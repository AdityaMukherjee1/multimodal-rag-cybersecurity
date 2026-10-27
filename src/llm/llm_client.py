"""LLM client factory for creating provider-specific LLM instances.

Use create_llm() to instantiate the appropriate LLM based on
configuration. New providers can be added by:
1. Creating a new class that extends BaseLLM
2. Registering it in the PROVIDERS dict below
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.llm.base_llm import BaseLLM
from src.llm.gemini_provider import GeminiProvider
from src.llm.mock_provider import MockLLMProvider
from src.llm.openai_provider import OpenAICompatibleProvider
from src.llm.placeholder_llm import PlaceholderLLM

logger = logging.getLogger(__name__)

# Registry of available LLM providers
PROVIDERS: dict[str, type[BaseLLM]] = {
    "placeholder": PlaceholderLLM,
    "mock": MockLLMProvider,
    "openai": OpenAICompatibleProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "google_gemini": GeminiProvider,
    "groq": OpenAICompatibleProvider,
    "ollama": OpenAICompatibleProvider,
}


def create_llm(provider: str = "placeholder", **kwargs: Any) -> BaseLLM:
    """Factory function to create an LLM instance.

    Args:
        provider: Name of the LLM provider ('gemini', 'groq', 'openai', 'ollama', 'mock', etc.).
        **kwargs: Additional arguments passed to the provider constructor.

    Returns:
        An instance of BaseLLM for the specified provider.
    """
    provider_lower = provider.lower().strip()

    if provider_lower not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS.keys()))
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Available providers: {available}. "
            f"To add a new provider, implement BaseLLM and register it in PROVIDERS."
        )

    # Smart default configurations for popular modern AI backends
    if provider_lower in ["gemini", "google", "google_gemini"]:
        kwargs.setdefault("base_url", "https://generativelanguage.googleapis.com/v1beta/openai/")
        if not kwargs.get("model_name"):
            kwargs["model_name"] = "gemini-3.8-flash"
        if not kwargs.get("api_key"):
            kwargs["api_key"] = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    elif provider_lower == "groq":
        kwargs.setdefault("base_url", "https://api.groq.com/openai/v1")
        kwargs.setdefault("model_name", "llama-3.3-70b-versatile")
        if not kwargs.get("api_key"):
            kwargs["api_key"] = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    elif provider_lower == "ollama":
        kwargs.setdefault("base_url", "http://localhost:11434/v1")
        kwargs.setdefault("model_name", "llama3.2")
        kwargs.setdefault("api_key", "ollama")

    llm_class = PROVIDERS[provider_lower]
    llm = llm_class(**kwargs)

    logger.info("Created LLM: provider='%s', class='%s'", provider, llm_class.__name__)
    return llm
