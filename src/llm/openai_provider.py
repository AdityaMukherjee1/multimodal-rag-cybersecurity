"""OpenAI-compatible LLM Provider.

Supports OpenAI (e.g. gpt-4o-mini), Groq, OpenRouter, LM Studio, Ollama,
or any OpenAI-compliant API endpoint using standard HTTP requests via httpx.
Credentials and endpoints are loaded exclusively from environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

from src.llm.base_llm import BaseLLM
from src.llm.output_parser import InvestigationOutputParser
from src.llm.prompt_templates import INVESTIGATION_SYSTEM_PROMPT, build_investigation_user_prompt
from src.llm.schema import LLMResponse, StructuredInvestigationAnswer

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseLLM):
    """Provider connecting to any OpenAI-compatible Chat Completions API.

    API keys and base URLs are resolved exclusively from environment variables:
    1. OPENAI_API_KEY (or LLM_API_KEY)
    2. OPENAI_BASE_URL (or LLM_BASE_URL, default: 'https://api.openai.com/v1')
    3. OPENAI_MODEL (or LLM_MODEL_NAME, default: 'gpt-4o-mini')
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_sec: float = 35.0,
    ) -> None:
        # Resolve credentials strictly without exposing in logs
        raw_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        if not raw_key and os.environ.get("LLM_API_KEY"):
            candidate = os.environ.get("LLM_API_KEY", "").strip()
            # Do not use Google Gemini keys for OpenAI endpoints
            if not candidate.startswith("AQ.") and not candidate.startswith("AIza"):
                raw_key = candidate
        self._api_key = raw_key.strip()

        raw_base = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1"
        self.base_url = raw_base.rstrip("/")

        self.model_name = (
            model_name
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("LLM_MODEL_NAME")
            or "gpt-4o-mini"
        ).strip()

        self.temperature = float(os.environ.get("LLM_TEMPERATURE", temperature))
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", max_tokens))
        self.timeout_sec = timeout_sec

        masked_key = f"...{self._api_key[-4:]}" if len(self._api_key) > 6 else ("SET" if self._api_key else "NOT_SET")
        logger.info(
            "OpenAICompatibleProvider initialized (model='%s', endpoint='%s', key=%s)",
            self.model_name,
            self.base_url,
            masked_key,
        )

    @property
    def is_configured(self) -> bool:
        """Check if an API key is available (or if calling a local unauthenticated endpoint)."""
        # Local endpoints like Ollama (localhost:11434) or LM Studio do not strictly require an API key
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            return True
        return bool(self._api_key)

    def generate(
        self,
        prompt: str,
        context: list[dict[str, Any]] | str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Call the Chat Completions endpoint and return raw text output."""
        response = self.generate_response(
            prompt=prompt,
            context=context,
            system_prompt=system_prompt,
            **kwargs,
        )
        if response.error:
            raise RuntimeError(response.error)
        return response.text

    def generate_response(
        self,
        prompt: str,
        context: list[dict[str, Any]] | str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Perform full completion request with timing, token accounting, and structured parsing."""
        if not self.is_configured:
            err_msg = (
                "LLM unavailable: OPENAI_API_KEY is not configured in .env or environment variables. "
                "Retrieved evidence is still available below."
            )
            logger.warning(err_msg)
            return LLMResponse(
                text="",
                structured=None,
                model=self.model_name,
                provider=self.provider_name,
                error=err_msg,
            )

        # Assemble prompt and context
        sys_p = system_prompt or INVESTIGATION_SYSTEM_PROMPT
        if isinstance(context, str):
            evidence_str = context
        elif isinstance(context, list):
            # Format list of dicts if raw list passed
            evidence_str = "\n\n".join([f"Evidence: {c.get('content', '')}" for c in context])
        else:
            evidence_str = "No context provided."

        user_content = build_investigation_user_prompt(prompt, evidence_str)

        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_content},
        ]

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

        # Override payload params if kwargs passed
        payload.update({k: v for k, v in kwargs.items() if k not in ["timeout"]})

        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self.base_url}/chat/completions"
        start_t = time.perf_counter()

        logger.info("Sending LLM request to %s (model=%s)", url, self.model_name)

        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                res = client.post(url, json=payload, headers=headers)
                latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

                if res.status_code == 400 and "response_format" in res.text.lower() and "response_format" in payload:
                    logger.warning("Target endpoint does not support 'response_format'; retrying without parameter.")
                    payload_fallback = dict(payload)
                    payload_fallback.pop("response_format", None)
                    res = client.post(url, json=payload_fallback, headers=headers)
                    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

                if res.status_code != 200:
                    err_detail = f"HTTP {res.status_code}: {res.text[:250]}"
                    # Sanitize error to prevent key leakage
                    if self._api_key and self._api_key in err_detail:
                        err_detail = err_detail.replace(self._api_key, "[REDACTED]")
                    logger.error("LLM API returned error: %s", err_detail)
                    return LLMResponse(
                        text="",
                        structured=None,
                        model=self.model_name,
                        provider=self.provider_name,
                        latency_ms=latency_ms,
                        error=f"LLM request failed: {err_detail}",
                    )

                data = res.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                raw_text = message.get("content", "")

                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                token_dict = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }

                logger.info(
                    "LLM response received in %.1fms (tokens: %d prompt, %d comp, %d total)",
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                )

                structured = InvestigationOutputParser.parse(
                    raw_text=raw_text,
                    model_name=self.model_name,
                    token_usage=token_dict,
                )

                return LLMResponse(
                    text=raw_text,
                    structured=structured,
                    model=self.model_name,
                    provider=self.provider_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    error=None,
                )

        except httpx.TimeoutException:
            latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
            err_msg = f"LLM request timed out after {self.timeout_sec:.0f} seconds. Retrieved evidence is still available below."
            logger.warning(err_msg)
            return LLMResponse(
                text="",
                structured=None,
                model=self.model_name,
                provider=self.provider_name,
                latency_ms=latency_ms,
                error=err_msg,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
            err_str = str(exc)
            if self._api_key and self._api_key in err_str:
                err_str = err_str.replace(self._api_key, "[REDACTED]")
            err_msg = f"LLM connection error: {err_str}. Retrieved evidence is still available below."
            logger.error("LLM execution error: %s", err_msg)
            return LLMResponse(
                text="",
                structured=None,
                model=self.model_name,
                provider=self.provider_name,
                latency_ms=latency_ms,
                error=err_msg,
            )
