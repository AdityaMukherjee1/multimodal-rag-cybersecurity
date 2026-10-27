"""Google Gemini Native LLM Provider.

Uses the official Google Generative Language API (v1beta generateContent)
for fast, native multimodal and text reasoning with structured schema support.
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


class GeminiProvider(BaseLLM):
    """Native Google Gemini API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout_sec: float = 40.0,
        **kwargs: Any,
    ) -> None:
        raw_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or ""
        )
        self._api_key = raw_key.strip()

        self.model_name = (
            model_name
            or os.environ.get("GEMINI_MODEL")
            or os.environ.get("LLM_MODEL_NAME")
            or "gemini-3.1-flash-lite-preview"
        ).strip()
        # Clean any 'models/' prefix if user provided it
        if self.model_name.startswith("models/"):
            self.model_name = self.model_name.replace("models/", "")
        if self.model_name in ["", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]:
            self.model_name = "gemini-3.1-flash-lite-preview"

        self.temperature = float(os.environ.get("LLM_TEMPERATURE", temperature))
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", max_tokens))
        self.timeout_sec = 60.0

        masked_key = f"...{self._api_key[-4:]}" if len(self._api_key) > 6 else ("SET" if self._api_key else "NOT_SET")
        logger.info(
            "GeminiProvider initialized (model='%s', key=%s)",
            self.model_name,
            masked_key,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def generate(
        self,
        prompt: str,
        context: list[dict[str, Any]] | str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        response = self.generate_response(
            prompt=prompt,
            context=context,
            system_prompt=system_prompt,
            **kwargs,
        )
        if response.error:
            raise RuntimeError(response.error)
        if response.structured and response.structured.answer:
            return response.structured.answer
        return response.text

    def generate_response(
        self,
        prompt: str,
        context: list[dict[str, Any]] | str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call Gemini generateContent and parse structured investigation response."""
        if not self.is_configured:
            err_msg = (
                "Gemini unavailable: GEMINI_API_KEY or LLM_API_KEY is not configured in .env. "
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

        sys_p = system_prompt or INVESTIGATION_SYSTEM_PROMPT
        if isinstance(context, str):
            evidence_str = context
        elif isinstance(context, list):
            evidence_str = "\n\n".join([f"Evidence: {c.get('content', '')}" for c in context])
        else:
            evidence_str = "No context provided."

        user_content = build_investigation_user_prompt(prompt, evidence_str)

        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": sys_p}]},
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
            },
        }

        # Allow kwargs override
        for k, v in kwargs.items():
            if k not in ["timeout"]:
                payload["generationConfig"][k] = v

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self._api_key}"
        start_t = time.perf_counter()

        logger.info("Sending Gemini request to %s (model=%s)", url.split("?")[0], self.model_name)

        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                res = client.post(url, json=payload, headers={"Content-Type": "application/json"})
                latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

                if res.status_code != 200:
                    err_detail = f"HTTP {res.status_code}: {res.text[:250]}"
                    if self._api_key and self._api_key in err_detail:
                        err_detail = err_detail.replace(self._api_key, "[REDACTED]")
                    logger.error("Gemini API returned error: %s", err_detail)
                    return LLMResponse(
                        text="",
                        structured=None,
                        model=self.model_name,
                        provider=self.provider_name,
                        latency_ms=latency_ms,
                        error=f"Gemini request failed: {err_detail}",
                    )

                data = res.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return LLMResponse(
                        text="",
                        structured=None,
                        model=self.model_name,
                        provider=self.provider_name,
                        latency_ms=latency_ms,
                        error="Gemini returned no candidates.",
                    )

                parts = candidates[0].get("content", {}).get("parts", [])
                raw_text = parts[0].get("text", "") if parts else ""

                usage = data.get("usageMetadata", {})
                prompt_tokens = usage.get("promptTokenCount", 0)
                completion_tokens = usage.get("candidatesTokenCount", 0)
                total_tokens = usage.get("totalTokenCount", 0)

                token_dict = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }

                logger.info(
                    "Gemini response received in %.1fms (tokens: %d prompt, %d comp, %d total)",
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
            err_msg = f"Gemini request timed out after {self.timeout_sec:.0f} seconds. Retrieved evidence is still available below."
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
            err_msg = f"Gemini connection error: {err_str}. Retrieved evidence is still available below."
            logger.error("Gemini execution error: %s", err_msg)
            return LLMResponse(
                text="",
                structured=None,
                model=self.model_name,
                provider=self.provider_name,
                latency_ms=latency_ms,
                error=err_msg,
            )
