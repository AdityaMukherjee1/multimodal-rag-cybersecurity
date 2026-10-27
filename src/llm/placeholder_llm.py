"""Placeholder LLM for offline testing.

Formats retrieved context into a structured response template
without calling any real LLM API. Useful for testing the
pipeline end-to-end without API keys or network access.
"""

from __future__ import annotations

import logging
from typing import Any

from src.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class PlaceholderLLM(BaseLLM):
    """Offline placeholder that formats context into a structured response.

    This does NOT perform any AI generation. It simply organizes the
    retrieved evidence into a readable template to verify that the
    retrieval pipeline is working correctly.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.model_name = "placeholder"

    def generate(
        self,
        prompt: str,
        context: list[dict[str, Any]] | None = None,
    ) -> str:
        """Format retrieved context into a structured response.

        Args:
            prompt: The user's question.
            context: List of retrieved evidence chunks.

        Returns:
            Formatted text summarizing the retrieved evidence.
        """
        logger.info("PlaceholderLLM generating response (no real LLM call)")

        lines: list[str] = [
            "=" * 60,
            "PLACEHOLDER LLM RESPONSE",
            "(No real LLM was called — this is a pipeline test)",
            "=" * 60,
            "",
            f"Question: {prompt}",
            "",
        ]

        if not context:
            lines.append("No evidence was retrieved for this question.")
        else:
            lines.append(f"Retrieved {len(context)} evidence chunk(s):\n")

            for i, chunk in enumerate(context, start=1):
                source = chunk.get("metadata", {}).get("filename", "unknown")
                score = chunk.get("score", 0.0)
                chunk_idx = chunk.get("metadata", {}).get("chunk_index", "?")
                content = chunk.get("content", "")

                # Truncate long content for display
                preview = content[:300] + "..." if len(content) > 300 else content

                lines.extend([
                    f"--- Evidence {i} ---",
                    f"  Source: {source}",
                    f"  Chunk: {chunk_idx}",
                    f"  Score: {score:.4f}",
                    f"  Content:\n    {preview}",
                    "",
                ])

            lines.extend([
                "-" * 60,
                "NOTE: Replace PlaceholderLLM with a real LLM provider",
                "(e.g., OpenAI, Google Gemini) to get AI-generated analysis.",
                "-" * 60,
            ])

        return "\n".join(lines)
