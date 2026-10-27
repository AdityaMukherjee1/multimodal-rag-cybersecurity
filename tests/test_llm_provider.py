"""Unit tests for LLM provider architecture and factory.

Ensures provider-agnostic abstraction, mock determinism, error handling,
and credential isolation without initiating live network calls.
"""

import os
from unittest.mock import patch

import pytest

from src.llm.base_llm import BaseLLM
from src.llm.llm_client import create_llm
from src.llm.mock_provider import MockLLMProvider
from src.llm.openai_provider import OpenAICompatibleProvider
from src.llm.placeholder_llm import PlaceholderLLM


class TestLLMProviderFactory:
    def test_create_placeholder_llm(self):
        llm = create_llm("placeholder")
        assert isinstance(llm, PlaceholderLLM)
        assert llm.provider_name == "PlaceholderLLM"

    def test_create_mock_llm(self):
        llm = create_llm("mock")
        assert isinstance(llm, MockLLMProvider)
        assert llm.provider_name == "MockLLMProvider"

    def test_create_openai_llm(self):
        llm = create_llm("openai")
        assert isinstance(llm, OpenAICompatibleProvider)
        assert llm.provider_name == "OpenAICompatibleProvider"

    def test_unknown_provider_raises_error(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm("nonexistent_provider_xyz")


class TestMockLLMProvider:
    def test_mock_generation_deterministic(self):
        provider = MockLLMProvider()
        context = "Forensic evidence [DOCX-001] indicates suspicious PowerShell execution."
        response = provider.generate_response(
            prompt="What evidence indicates PowerShell bypass?",
            context=context,
        )

        assert response.is_success
        assert response.structured is not None
        assert response.structured.is_parsed_json
        assert len(response.structured.claims) > 0
        assert "DOCX-001" in response.structured.claims[0].supporting_evidence_ids
        assert "T1059.001" in response.structured.attck_techniques
        assert response.total_tokens > 0

    def test_mock_simulated_error(self):
        provider = MockLLMProvider(simulate_error=True, error_message="Simulated connection timeout")
        response = provider.generate_response(
            prompt="Investigate credential dumping",
            context="Some evidence",
        )

        assert not response.is_success
        assert "Simulated connection timeout" in response.error
        assert response.structured is None

    def test_mock_simulated_malformed_json(self):
        provider = MockLLMProvider(simulate_malformed_json=True)
        response = provider.generate_response(
            prompt="Investigate CVE-2021-44228",
            context="[EVIDENCE-001] Log4j advisory",
        )

        assert response.is_success
        assert response.structured is not None
        # Heuristic fallback handles malformed JSON without crashing
        assert not response.structured.is_parsed_json
        assert len(response.structured.claims) > 0


class TestOpenAICompatibleProvider:
    def test_unconfigured_key_returns_graceful_error(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "LLM_API_KEY": ""}, clear=False):
            provider = OpenAICompatibleProvider(api_key="", base_url="https://api.openai.com/v1")
            assert not provider.is_configured
            response = provider.generate_response(
                prompt="Investigate alert",
                context="Evidence text",
            )
            assert not response.is_success
            assert "OPENAI_API_KEY is not configured" in response.error
            assert "Retrieved evidence is still available below" in response.error

    def test_configured_local_endpoint_without_key(self):
        provider = OpenAICompatibleProvider(base_url="http://localhost:11434/v1")
        assert provider.is_configured

    def test_api_key_not_exposed_in_repr_or_logs(self):
        secret_key = "sk-secret1234567890abcdef"
        provider = OpenAICompatibleProvider(api_key=secret_key)
        assert secret_key not in str(provider)
        assert secret_key not in repr(provider)
