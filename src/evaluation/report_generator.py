"""Report and Plot Generator for Phase 5 Empirical Evaluation.

Generates:
1. Machine-readable JSON execution records and CSV summary tables.
2. Publication-quality Matplotlib evaluation figures.
3. Statistical significance analysis.
4. Markdown tables and documentation blocks for research reporting.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.evaluation.configurations import ExperimentRunOutput

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "evaluation" / "plots"


def aggregate_config_metrics(runs: list[ExperimentRunOutput]) -> dict[str, Any]:
    """Calculate descriptive statistics for a list of runs belonging to one configuration."""
    if not runs:
        return {}

    def _mean(values: list[float]) -> float:
        return round(float(np.mean(values)), 4) if values else 0.0

    def _std(values: list[float]) -> float:
        return round(float(np.std(values)), 4) if values else 0.0

    r1 = [r.retrieval_metrics.get("recall@1", 0.0) for r in runs]
    r3 = [r.retrieval_metrics.get("recall@3", 0.0) for r in runs]
    r5 = [r.retrieval_metrics.get("recall@5", 0.0) for r in runs]
    r10 = [r.retrieval_metrics.get("recall@10", 0.0) for r in runs]
    mrr = [r.retrieval_metrics.get("mrr", 0.0) for r in runs]

    ans_acc = [r.answer_accuracy for r in runs]
    cit_acc = [r.citation_accuracy for r in runs]
    grounding = [r.grounding_score for r in runs]
    hallucination = [r.hallucination_rate for r in runs]

    ret_lat = [r.retrieval_latency_ms for r in runs]
    llm_lat = [r.llm_latency_ms for r in runs]
    verif_lat = [r.verification_latency_ms for r in runs]
    tot_lat = [r.total_latency_ms for r in runs]
    tokens = [r.total_tokens for r in runs]
    vram = [r.gpu_vram_mb for r in runs]

    return {
        "count": len(runs),
        "recall@1_mean": _mean(r1),
        "recall@1_std": _std(r1),
        "recall@3_mean": _mean(r3),
        "recall@3_std": _std(r3),
        "recall@5_mean": _mean(r5),
        "recall@5_std": _std(r5),
        "recall@10_mean": _mean(r10),
        "mrr_mean": _mean(mrr),
        "mrr_std": _std(mrr),
        "answer_accuracy_mean": _mean(ans_acc),
        "answer_accuracy_std": _std(ans_acc),
        "citation_accuracy_mean": _mean(cit_acc),
        "grounding_score_mean": _mean(grounding),
        "grounding_score_std": _std(grounding),
        "hallucination_rate_mean": _mean(hallucination),
        "hallucination_rate_std": _std(hallucination),
        "retrieval_latency_ms_mean": _mean(ret_lat),
        "llm_latency_ms_mean": _mean(llm_lat),
        "verification_latency_ms_mean": _mean(verif_lat),
        "total_latency_ms_mean": _mean(tot_lat),
        "total_tokens_mean": _mean(tokens),
        "gpu_vram_mb_mean": _mean(vram),
    }


def compute_statistical_tests(
    config_a_scores: list[float],
    config_b_scores: list[float],
    metric_name: str = "metric",
) -> dict[str, Any]:
    """Execute paired t-test and Wilcoxon signed-rank test between two configurations."""
    if len(config_a_scores) != len(config_b_scores) or len(config_a_scores) < 5:
        return {
            "metric": metric_name,
            "sample_size": len(config_a_scores),
            "is_valid": False,
            "reason": "Insufficient sample size (N < 5) or unequal array lengths",
        }

    diffs = np.array(config_a_scores) - np.array(config_b_scores)
    if np.all(diffs == 0):
        return {
            "metric": metric_name,
            "sample_size": len(config_a_scores),
            "is_valid": True,
            "p_value_ttest": 1.0,
            "p_value_wilcoxon": 1.0,
            "is_statistically_significant": False,
            "note": "Identical distributions (zero differences).",
        }

    try:
        t_stat, p_ttest = stats.ttest_rel(config_a_scores, config_b_scores)
        p_ttest_val = round(float(p_ttest), 5)
    except Exception as e:
        t_stat, p_ttest_val = 0.0, 1.0

    try:
        w_stat, p_wilcoxon = stats.wilcoxon(config_a_scores, config_b_scores)
        p_wilcoxon_val = round(float(p_wilcoxon), 5)
    except Exception as e:
        w_stat, p_wilcoxon_val = 0.0, 1.0

    is_sig = bool(p_ttest_val < 0.05 or p_wilcoxon_val < 0.05)

    return {
        "metric": metric_name,
        "sample_size": len(config_a_scores),
        "is_valid": True,
        "p_value_ttest": p_ttest_val,
        "p_value_wilcoxon": p_wilcoxon_val,
        "is_statistically_significant": is_sig,
    }


def export_results_to_disk(
    all_runs: list[ExperimentRunOutput],
    results_dir: Path | None = None,
) -> dict[str, Path]:
    """Export complete runs to JSON and summarized tables to CSV."""
    r_dir = results_dir or DEFAULT_RESULTS_DIR
    r_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw JSON
    json_path = r_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in all_runs], f, indent=2)

    # Group runs by configuration
    grouped: dict[str, list[ExperimentRunOutput]] = {}
    for r in all_runs:
        grouped.setdefault(r.config_name, []).append(r)

    # 2. Baseline Comparison CSV
    baseline_keys = ["A_LLM_Only", "B_Text_RAG", "C_Multimodal_RAG", "D_Full_System"]
    csv_path = r_dir / "baseline_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Configuration", "N", "Recall@1", "Recall@3", "Recall@5", "MRR",
            "Answer_Acc", "Citation_Acc", "Grounding_Score", "Hallucination_Rate",
            "Ret_Latency_ms", "LLM_Latency_ms", "Verif_Latency_ms", "Total_Latency_ms",
        ])
        for k in baseline_keys:
            if k in grouped:
                agg = aggregate_config_metrics(grouped[k])
                writer.writerow([
                    k, agg["count"],
                    f"{agg['recall@1_mean']:.4f}", f"{agg['recall@3_mean']:.4f}", f"{agg['recall@5_mean']:.4f}", f"{agg['mrr_mean']:.4f}",
                    f"{agg['answer_accuracy_mean']:.4f}", f"{agg['citation_accuracy_mean']:.4f}",
                    f"{agg['grounding_score_mean']:.4f}", f"{agg['hallucination_rate_mean']:.4f}",
                    f"{agg['retrieval_latency_ms_mean']:.1f}", f"{agg['llm_latency_ms_mean']:.1f}",
                    f"{agg['verification_latency_ms_mean']:.1f}", f"{agg['total_latency_ms_mean']:.1f}",
                ])

    # 3. Ablation Comparison CSV
    ablation_keys = [
        "D_Full_System",
        "Ablation_WO_Visual",
        "Ablation_WO_Fusion",
        "Ablation_WO_Verification",
        "Ablation_WO_KnowledgeBase",
    ]
    ablation_csv_path = r_dir / "ablation_results.csv"
    with open(ablation_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ablation_Setting", "N", "Recall@5", "MRR", "Grounding_Score",
            "Hallucination_Rate", "Answer_Accuracy", "Total_Latency_ms",
        ])
        for k in ablation_keys:
            if k in grouped:
                agg = aggregate_config_metrics(grouped[k])
                writer.writerow([
                    k, agg["count"],
                    f"{agg['recall@5_mean']:.4f}", f"{agg['mrr_mean']:.4f}",
                    f"{agg['grounding_score_mean']:.4f}", f"{agg['hallucination_rate_mean']:.4f}",
                    f"{agg['answer_accuracy_mean']:.4f}", f"{agg['total_latency_ms_mean']:.1f}",
                ])

    return {
        "json": json_path,
        "baseline_csv": csv_path,
        "ablation_csv": ablation_csv_path,
    }


def generate_evaluation_plots(
    all_runs: list[ExperimentRunOutput],
    plots_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate high-resolution comparative evaluation figures using Matplotlib."""
    p_dir = plots_dir or DEFAULT_PLOTS_DIR
    p_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[ExperimentRunOutput]] = {}
    for r in all_runs:
        grouped.setdefault(r.config_name, []).append(r)

    generated_plots = {}

    # 1. Retrieval Performance Comparison Plot (Recall@K & MRR)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ret_configs = ["B_Text_RAG", "C_Multimodal_RAG", "D_Full_System"]
    labels = ["Text RAG (Config B)", "Multimodal RAG (Config C)", "Full Proposed (Config D)"]

    r1_vals = [aggregate_config_metrics(grouped.get(c, []))["recall@1_mean"] for c in ret_configs]
    r3_vals = [aggregate_config_metrics(grouped.get(c, []))["recall@3_mean"] for c in ret_configs]
    r5_vals = [aggregate_config_metrics(grouped.get(c, []))["recall@5_mean"] for c in ret_configs]
    mrr_vals = [aggregate_config_metrics(grouped.get(c, []))["mrr_mean"] for c in ret_configs]

    x = np.arange(len(ret_configs))
    width = 0.2

    ax.bar(x - 1.5 * width, r1_vals, width, label="Recall@1", color="#4e79a7")
    ax.bar(x - 0.5 * width, r3_vals, width, label="Recall@3", color="#f28e2b")
    ax.bar(x + 0.5 * width, r5_vals, width, label="Recall@5", color="#e15759")
    ax.bar(x + 1.5 * width, mrr_vals, width, label="MRR", color="#76b7b2")

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_title("Retrieval Performance across System Architectures", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    ret_plot_path = p_dir / "retrieval_performance.png"
    plt.tight_layout()
    plt.savefig(ret_plot_path)
    plt.close(fig)
    generated_plots["retrieval_plot"] = ret_plot_path

    # 2. Grounding vs Hallucination Rate Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    all_configs = ["A_LLM_Only", "B_Text_RAG", "C_Multimodal_RAG", "D_Full_System"]
    c_labels = ["LLM Only (A)", "Text RAG (B)", "Multimodal (C)", "Full System (D)"]

    grounding_vals = [aggregate_config_metrics(grouped.get(c, []))["grounding_score_mean"] * 100 for c in all_configs]
    halluc_vals = [aggregate_config_metrics(grouped.get(c, []))["hallucination_rate_mean"] * 100 for c in all_configs]

    x = np.arange(len(all_configs))
    width = 0.35

    ax.bar(x - width / 2, grounding_vals, width, label="Grounding Score (%)", color="#59a14f")
    ax.bar(x + width / 2, halluc_vals, width, label="Hallucination Rate (%)", color="#e15759")

    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Grounding Rate vs. Hallucination Reduction across Baselines", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(c_labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    for i in range(len(all_configs)):
        ax.text(x[i] - width / 2, grounding_vals[i] + 2, f"{grounding_vals[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(x[i] + width / 2, halluc_vals[i] + 2, f"{halluc_vals[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")

    grounding_plot_path = p_dir / "grounding_vs_hallucination.png"
    plt.tight_layout()
    plt.savefig(grounding_plot_path)
    plt.close(fig)
    generated_plots["grounding_plot"] = grounding_plot_path

    # 3. Latency Breakdown Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ret_lat = [aggregate_config_metrics(grouped.get(c, []))["retrieval_latency_ms_mean"] for c in all_configs]
    llm_lat = [aggregate_config_metrics(grouped.get(c, []))["llm_latency_ms_mean"] for c in all_configs]
    verif_lat = [aggregate_config_metrics(grouped.get(c, []))["verification_latency_ms_mean"] for c in all_configs]

    ax.bar(c_labels, ret_lat, label="Retrieval (ms)", color="#4e79a7")
    ax.bar(c_labels, llm_lat, bottom=ret_lat, label="LLM Reasoning (ms)", color="#f28e2b")
    ax.bar(c_labels, verif_lat, bottom=np.array(ret_lat) + np.array(llm_lat), label="Verification (ms)", color="#59a14f")

    ax.set_ylabel("Latency (milliseconds)", fontsize=11)
    ax.set_title("End-to-End Latency Breakdown Across Architectural Configurations", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    latency_plot_path = p_dir / "latency_breakdown.png"
    plt.tight_layout()
    plt.savefig(latency_plot_path)
    plt.close(fig)
    generated_plots["latency_plot"] = latency_plot_path

    # 4. Ablation Study Impact Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ablation_configs = [
        "D_Full_System",
        "Ablation_WO_Visual",
        "Ablation_WO_Fusion",
        "Ablation_WO_Verification",
        "Ablation_WO_KnowledgeBase",
    ]
    abl_labels = [
        "Full System",
        "W/O Visual",
        "W/O Fusion",
        "W/O Verification",
        "W/O Threat Intel",
    ]

    abl_grounding = [aggregate_config_metrics(grouped.get(c, []))["grounding_score_mean"] * 100 for c in ablation_configs]
    abl_recall5 = [aggregate_config_metrics(grouped.get(c, []))["recall@5_mean"] * 100 for c in ablation_configs]

    x = np.arange(len(ablation_configs))
    width = 0.35

    ax.bar(x - width / 2, abl_grounding, width, label="Grounding Score (%)", color="#2b5c8f")
    ax.bar(x + width / 2, abl_recall5, width, label="Recall@5 (%)", color="#d95f02")

    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Ablation Study: Component Removal Impact on Retrieval & Grounding", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(abl_labels, fontsize=10, rotation=15)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")

    for i in range(len(ablation_configs)):
        ax.text(x[i] - width / 2, abl_grounding[i] + 2, f"{abl_grounding[i]:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.text(x[i] + width / 2, abl_recall5[i] + 2, f"{abl_recall5[i]:.1f}%", ha="center", fontsize=8, fontweight="bold")

    ablation_plot_path = p_dir / "ablation_impact.png"
    plt.tight_layout()
    plt.savefig(ablation_plot_path)
    plt.close(fig)
    generated_plots["ablation_plot"] = ablation_plot_path

    return generated_plots
