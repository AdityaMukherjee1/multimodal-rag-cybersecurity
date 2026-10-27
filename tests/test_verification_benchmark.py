"""Unit tests for Phase 4 Verification Benchmark dataset and evaluation runner."""

from src.verification.evidence_verifier import HybridEvidenceVerifier
from src.verification.verification_benchmark import evaluate_verifier_on_benchmark, load_verification_benchmark


class TestVerificationBenchmark:
    def test_load_benchmark_dataset(self):
        cases = load_verification_benchmark()
        assert len(cases) >= 5
        categories = {c["category"] for c in cases}
        assert "supported_text_claim" in categories
        assert "supported_multimodal_claim" in categories
        assert "unsupported_hallucination" in categories
        assert "contradicted_claim" in cases[4]["category"] or "contradicted_claim" in categories

    def test_evaluate_hybrid_verifier_on_benchmark(self):
        verifier = HybridEvidenceVerifier()
        metrics = evaluate_verifier_on_benchmark(verifier=verifier)

        assert metrics["total_test_cases"] >= 5
        assert metrics["accuracy"] >= 0.80, f"Accuracy too low: {metrics['accuracy']}"
        assert metrics["hallucination_detection_rate"] >= 0.80, f"Hallucination detection too low: {metrics['hallucination_detection_rate']}"

        # Ensure all individual test cases have evaluation entries
        for r in metrics["results"]:
            assert "test_id" in r
            assert "actual_status" in r
            assert "reasoning" in r
