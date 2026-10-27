"""LLM client module with modular provider abstraction."""

from src.llm.base_llm import BaseLLM
from src.llm.context_builder import EvidenceContextBuilder, FormattedEvidenceContext
from src.llm.llm_client import create_llm
from src.llm.mock_provider import MockLLMProvider
from src.llm.openai_provider import OpenAICompatibleProvider
from src.llm.output_parser import InvestigationOutputParser
from src.llm.prompt_templates import INVESTIGATION_SYSTEM_PROMPT, build_investigation_user_prompt
from src.llm.schema import Claim, LLMResponse, StructuredInvestigationAnswer

__all__ = [
    "BaseLLM",
    "create_llm",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "EvidenceContextBuilder",
    "FormattedEvidenceContext",
    "InvestigationOutputParser",
    "INVESTIGATION_SYSTEM_PROMPT",
    "build_investigation_user_prompt",
    "Claim",
    "StructuredInvestigationAnswer",
    "LLMResponse",
]
