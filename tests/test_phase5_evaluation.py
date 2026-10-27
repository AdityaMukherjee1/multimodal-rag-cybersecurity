"""Unit and Integration Tests for Phase 5 Empirical Evaluation Subsystem.

Tests:
1. Benchmark dataset loading and schema validation.
2. Retrieval metrics computation (Recall@K, MRR).
3. Answer accuracy and citation precision computation.
4. Claim verification metrics (Precision, Recall, F1, Hallucination Rate).
5. Statistical significance calculation.
6. Execution of Configurations A, B, C, D on sample queries.
7. Execution of Ablations on sample queries.
8. Automated plot and report generation.
"""

from pathlib import Path

import pytest

from src.config.settings import Settings
from src.embeddings.embedding_service import EmbeddingService
from src.evaluation.ablation import AblationStudyRunner
from src.evaluation.configurations import EvaluationPipelineManager, ExperimentRunOutput
from src.evaluation.eval_runner import load_benchmark
from src.evaluation.metrics import (
    calculate_answer_accuracy,
    calculate_citation_accuracy,
    calculate_retrieval_metrics,
    calculate_verification_metrics,
    is_evidence_match,
)
from src.evaluation.report_generator import (
    aggregate_config_metrics,
    compute_statistical_tests,
    generate_evaluation_plots,
)
from src.llm.context_builder import EvidenceContextBuilder
from src.llm.mock_provider import MockLLMProvider
from src.multimodal.multimodal_retriever import MultimodalRetriever
from src.multimodal.visual_embedding_service import CLIPVisualEmbeddingService
from src.multimodal.visual_vector_store import VisualVectorStore
from src.retrieval.vector_store import VectorStore
from src.verification.evidence_verifier import HybridEvidenceVerifier
from src.verification.schema import VerificationStatus

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase5Evaluation:
    @classmethod
    def setup_class(cls):
        cls.settings = Settings()
        cls.text_store_dir = PROJECT_ROOT / "data" / "processed" / "faiss_index"
        cls.visual_store_dir = PROJECT_ROOT / "data" / "processed" / "visual_faiss_index"

        cls.text_store = VectorStore.load(cls.text_store_dir, dimension=384)
        cls.visual_store = VisualVectorStore.load(cls.visual_store_dir, dimension=512)
        cls.text_embed = EmbeddingService(cls.settings.embedding_model_name)
        cls.visual_embed = CLIPVisualEmbeddingService()

        cls.retriever = MultimodalRetriever(
            text_store=cls.text_store,
            visual_store=cls.visual_store,
            text_embedding_service=cls.text_embed,
            visual_embedding_service=cls.visual_embed,
        )
        cls.context_builder = EvidenceContextBuilder()
        cls.mock_llm = MockLLMProvider()
        cls.verifier = HybridEvidenceVerifier()

        cls.manager = EvaluationPipelineManager(
            retriever=cls.retriever,
            llm=cls.mock_llm,
            context_builder=cls.context_builder,
            verifier=cls.verifier,
        )
        cls.ablation_runner = AblationStudyRunner(manager=cls.manager)

    def test_benchmark_loading_and_schema(self):
        queries = load_benchmark()
        assert len(queries) == 24
        categories = {q["category"] for q in queries}
        assert len(categories) == 6
        assert "general_cybersecurity" in categories
        assert "incident_investigation" in categories
        assert "attck_cve_knowledge" in categories
        assert "visual_evidence" in categories
        assert "cross_modal_corroboration" in categories
        assert "adversarial_unsupported" in categories

        for q in queries:
            assert "query_id" in q
            assert "query" in q
            assert "ground_truth_evidence_ids" in q
            assert "key_reference_facts" in q
            assert "expected_grounding_status" in q

    def test_evidence_id_matching(self):
        assert is_evidence_match("ATTCK-T1059_001", "ATTCK-T1059_001")
        assert is_evidence_match("ATTCK-T1059_001", "T1059.001")
        assert is_evidence_match("CVE-CVE_2021_44228", "CVE-2021-44228")
        assert is_evidence_match("DOCX-soc_incident_investigation_briefing-003", "soc_incident_investigation_briefing.docx")

    def test_retrieval_metrics(self):
        retrieved = ["DOCX-001", "ATTCK-T1059_001", "IMG-001", "NET-002", "EVTX-003"]
        gt = ["ATTCK-T1059_001", "IMG-001"]

        metrics = calculate_retrieval_metrics(retrieved, gt, k_values=[1, 3, 5])
        assert metrics["recall@1"] == 0.0  # neither in top 1
        assert metrics["recall@3"] == 1.0  # both in top 3
        assert metrics["recall@5"] == 1.0
        assert metrics["mrr"] == 0.5  # first hit at rank 2 (1/2 = 0.5)

    def test_answer_accuracy_metric(self):
        ans = "The attacker used powershell.exe with -ExecutionPolicy Bypass on host WORKSTATION-01."
        facts = ["powershell.exe", "bypass", "workstation-01"]
        acc = calculate_answer_accuracy(ans, facts)
        assert acc == 1.0

        ans_partial = "Powershell execution was observed."
        acc_partial = calculate_answer_accuracy(ans_partial, facts)
        assert 0.0 < acc_partial < 1.0

    def test_citation_accuracy_metric(self):
        cited = ["ATTCK-T1059_001", "DOCX-001", "FAKE-999"]
        retrieved = ["ATTCK-T1059_001", "DOCX-001", "NET-002"]
        gt = ["ATTCK-T1059_001"]

        acc = calculate_citation_accuracy(cited, retrieved, gt)
        assert acc == round(2 / 3, 4)

    def test_verification_metrics(self):
        pred = [
            VerificationStatus.SUPPORTED,
            VerificationStatus.UNSUPPORTED,
            VerificationStatus.CONTRADICTED,
            VerificationStatus.SUPPORTED,
        ]
        exp = [
            VerificationStatus.SUPPORTED,
            VerificationStatus.UNSUPPORTED,
            VerificationStatus.CONTRADICTED,
            VerificationStatus.UNSUPPORTED,  # False Negative
        ]
        metrics = calculate_verification_metrics(pred, exp)
        assert metrics["accuracy"] == 0.75
        assert metrics["precision"] == 1.0  # All predicted hallucinations were true hallucinations
        assert metrics["recall"] == round(2 / 3, 4)
        assert metrics["f1"] > 0.70

    def test_statistical_tests(self):
        scores_a = [0.9, 0.85, 0.95, 0.88, 0.92, 0.89]
        scores_b = [0.6, 0.55, 0.65, 0.58, 0.62, 0.59]

        res = compute_statistical_tests(scores_a, scores_b, "Grounding")
        assert res["is_valid"] is True
        assert res["is_statistically_significant"] is True
        assert res["p_value_ttest"] < 0.05

    def test_config_a_execution(self):
        out = self.manager.run_config_a_llm_only("What is T1059?", "TEST-Q1", "test")
        assert out.config_name == "A_LLM_Only"
        assert len(out.retrieved_evidence_ids) == 0
        assert out.grounding_score == 0.0

    def test_config_b_execution(self):
        out = self.manager.run_config_b_text_rag("What is CVE-2021-44228?", "TEST-Q2", "test", top_k=3)
        assert out.config_name == "B_Text_RAG"
        assert len(out.retrieved_evidence_ids) == 3
        assert out.retrieval_latency_ms > 0

    def test_config_c_execution(self):
        out = self.manager.run_config_c_multimodal_rag("Show terminal screenshot of Mimikatz", "TEST-Q3", "test", top_k=3)
        assert out.config_name == "C_Multimodal_RAG"
        assert len(out.retrieved_evidence_ids) == 3

    def test_config_d_execution(self):
        out = self.manager.run_config_d_full_system("What evidence indicates credential dumping?", "TEST-Q4", "test", top_k=3)
        assert out.config_name == "D_Full_System"
        assert len(out.retrieved_evidence_ids) == 3
        assert out.verification_report is not None
        assert out.grounding_score > 0

    def test_ablation_executions(self):
        q = "What evidence demonstrates unauthorized PowerShell execution policy bypass?"
        out_no_vis = self.ablation_runner.run_ablation_no_visual(q, "TEST-ABL-1", "test", top_k=3)
        assert out_no_vis.config_name == "Ablation_WO_Visual"

        out_no_fus = self.ablation_runner.run_ablation_no_fusion(q, "TEST-ABL-2", "test", top_k=3)
        assert out_no_fus.config_name == "Ablation_WO_Fusion"

        out_no_ver = self.ablation_runner.run_ablation_no_verification(q, "TEST-ABL-3", "test", top_k=3)
        assert out_no_ver.config_name == "Ablation_WO_Verification"

        out_no_kb = self.ablation_runner.run_ablation_no_knowledge_base(q, "TEST-ABL-4", "test", top_k=3)
        assert out_no_kb.config_name == "Ablation_WO_KnowledgeBase"

    def test_plot_generation(self, tmp_path):
        q = "What evidence demonstrates unauthorized PowerShell execution policy bypass?"
        runs = [
            self.manager.run_config_a_llm_only(q, "Q1", "test"),
            self.manager.run_config_b_text_rag(q, "Q1", "test", top_k=2),
            self.manager.run_config_c_multimodal_rag(q, "Q1", "test", top_k=2),
            self.manager.run_config_d_full_system(q, "Q1", "test", top_k=2),
            self.ablation_runner.run_ablation_no_visual(q, "Q1", "test", top_k=2),
            self.ablation_runner.run_ablation_no_fusion(q, "Q1", "test", top_k=2),
            self.ablation_runner.run_ablation_no_verification(q, "Q1", "test", top_k=2),
            self.ablation_runner.run_ablation_no_knowledge_base(q, "Q1", "test", top_k=2),
        ]
        plots = generate_evaluation_plots(runs, plots_dir=tmp_path)
        assert len(plots) == 4
        for p in plots.values():
            assert p.exists()
            assert p.stat().st_size > 1000
