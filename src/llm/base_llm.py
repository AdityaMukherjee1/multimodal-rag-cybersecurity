"""Abstract base class for LLM providers.

Defines the interface that all LLM implementations must follow,
enabling provider-agnostic usage in the RAG pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.llm.schema import LLMResponse, StructuredInvestigationAnswer


class BaseLLM(ABC):
    """Abstract interface for language model providers.

    All LLM implementations (OpenAI, Mock, local models, etc.)
    must implement this interface. This ensures the pipeline is
    not coupled to any specific LLM provider.

    Subclasses must implement:
        - generate(prompt, context=..., system_prompt=..., **kwargs) -> str
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: list[dict[str, Any]] | str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response given a prompt and optional retrieved context.

        Args:
            prompt: The user's question or instruction.
            context: Optional list of retrieved evidence chunks or formatted evidence string.
            system_prompt: Optional system prompt to steer model behavior.
            **kwargs: Provider-specific runtime parameters.

        Returns:
            Generated text response.
        """
        ...

    def generate_response(
        self,
        prompt: str,
        context: list[dict[str, Any]] | str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a full LLMResponse object including token metrics.

        Default implementation wraps generate(). Providers can override
        for native token accounting and structured response parsing.
        """
        raw_text = self.generate(
            prompt=prompt,
            context=context,
            system_prompt=system_prompt,
            **kwargs,
        )
        return LLMResponse(
            text=raw_text,
            model=getattr(self, "model_name", "unknown"),
            provider=self.provider_name,
        )

    @property
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        return self.__class__.__name__
