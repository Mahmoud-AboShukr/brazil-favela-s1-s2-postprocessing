#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
07_summarize_frozen_probe_results_224.py

Summarize Criterion 1:
    Downstream frozen-probe performance for RTC vs SNAP-GRD.

This script reads the fold-specific outputs produced by:

    src/croma_probing/06_train_frozen_probe_224.py

Expected inputs:

    frozen_probe_aggregate_leave_one_region_ps224_st112_cover.csv
    frozen_probe_aggregate_leave_one_city_ps224_st112_cover.csv

    frozen_probe_rtc_vs_snap_comparison_leave_one_region_ps224_st112_cover.csv
    frozen_probe_rtc_vs_snap_comparison_leave_one_city_ps224_st112_cover.csv

It produces one consolidated Criterion 1 report:

    criterion1_frozen_probe_best_modalities_ps224_st112_cover.csv
    criterion1_rtc_vs_snap_primary_summary_ps224_st112_cover.csv
    criterion1_rtc_vs_snap_all_metrics_ps224_st112_cover.csv
    criterion1_frozen_probe_summary_ps224_st112_cover.json
    criterion1_frozen_probe_summary_ps224_st112_cover.md

Optional figures are also produced if matplotlib is installed:

    criterion1_average_precision_rtc_vs_snap_sar_only_ps224_st112_cover.png
    criterion1_average_precision_rtc_vs_snap_joint_ps224_st112_cover.png

Primary decision metric:
    Average Precision

Supporting metrics:
    ROC-AUC
    F1
    Precision
    Recall
    Balanced Accuracy
    Accuracy

Example:

python src/croma_probing/07_summarize_frozen_probe_results_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    log("ERROR", message)
    raise SystemExit(exit_code)


def path_to_str(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return str(path).replace("\\", "/")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if text == "":
            return default
        out = float(text)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def round_float(value: float, digits: int = 8) -> float:
    return round(safe_float(value, 0.0), digits)


def normalize_fold_mode(value: str) -> str:
    return str(value).strip()


def pretty_fold_mode(value: str) -> str:
    value = normalize_fold_mode(value)
    if value == "leave_one_region":
        return "Leave-one-region"
    if value == "leave_one_city":
        return "Leave-one-city"
    return value


def pretty_probe(value: str) -> str:
    mapping = {
        "logistic_regression": "Logistic regression",
        "linear_svm": "Linear SVM",
        "small_mlp": "Small MLP",
    }
    return mapping.get(str(value), str(value))


def pretty_comparison(value: str) -> str:
    mapping = {
        "sar_only_rtc_vs_snap": "SAR-only RTC vs SNAP-GRD",
        "joint_rtc_vs_snap": "Joint S2+S1 RTC vs SNAP-GRD",
    }
    return mapping.get(str(value), str(value))


def pretty_modality(value: str) -> str:
    mapping = {
        "s2": "S2 only",
        "s1_snap_vv_vh": "S1 SNAP-GRD VV/VH",
        "s1_rtc_vv_vh": "S1 RTC VV/VH",
        "s2_s1_snap_vv_vh": "S2 + S1 SNAP-GRD VV/VH",
        "s2_s1_rtc_vv_vh": "S2 + S1 RTC VV/VH",
    }
    return mapping.get(str(value), str(value))


# ---------------------------------------------------------------------
# CSV / JSON / Markdown I/O
# ---------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        fail(f"Required input CSV does not exist: {path_to_str(path)}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        fail(f"Input CSV is empty: {path_to_str(path)}")

    return rows


def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    overwrite: bool,
    fieldnames: Optional[List[str]] = None,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    if fieldnames is None:
        if not rows:
            fail(f"No rows to write and no fieldnames provided for: {path_to_str(path)}")
        fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(
    path: Path,
    *,
    summary: Dict[str, object],
    best_rows: List[Dict[str, object]],
    primary_rows: List[Dict[str, object]],
    decision_rows: List[Dict[str, object]],
    output_paths: Dict[str, Path],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Criterion 1 Summary: Frozen CROMA Probe Performance")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Frozen-probe directory: `{summary['frozen_probe_dir']}`")
    lines.append(f"- Fold modes summarized: `{';'.join(summary['fold_modes'])}`")
    lines.append(f"- Primary metric: `{summary['primary_metric']}`")
    lines.append("")

    lines.append("### Main conclusion")
    lines.append("")
    lines.append(summary["main_conclusion"])
    lines.append("")

    lines.append("### Decision by comparison")
    lines.append("")
    lines.append("| comparison | decision | evidence summary |")
    lines.append("|---|---|---|")
    for row in decision_rows:
        lines.append(
            f"| {row['comparison_pretty']} | "
            f"{row['decision']} | "
            f"{row['evidence_summary']} |"
        )

    lines.append("")
    lines.append("## Best modalities by Average Precision")
    lines.append("")
    lines.append("| fold mode | probe | best modality | AP mean | second modality | second AP mean | margin |")
    lines.append("|---|---|---|---:|---|---:|---:|")

    for row in best_rows:
        lines.append(
            f"| {row['fold_mode_pretty']} | "
            f"{row['probe_pretty']} | "
            f"{row['best_modality_pretty']} | "
            f"{row['best_average_precision_mean']} | "
            f"{row['second_modality_pretty']} | "
            f"{row['second_average_precision_mean']} | "
            f"{row['margin_best_minus_second']} |"
        )

    lines.append("")
    lines.append("## RTC vs SNAP-GRD: Average Precision")
    lines.append("")
    lines.append("| comparison | fold mode | probe | SNAP mean | RTC mean | delta RTC-SNAP | RTC wins | SNAP wins | ties | decision |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")

    for row in primary_rows:
        lines.append(
            f"| {row['comparison_pretty']} | "
            f"{row['fold_mode_pretty']} | "
            f"{row['probe_pretty']} | "
            f"{row['snap_mean']} | "
            f"{row['rtc_mean']} | "
            f"{row['delta_rtc_minus_snap']} | "
            f"{row['rtc_wins']} | "
            f"{row['snap_wins']} | "
            f"{row['ties']} | "
            f"{row['direction']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Average Precision is the primary metric because the positive favela class can be unevenly distributed across spatial folds. A higher Average Precision indicates that the embedding and probe combination ranks favela-positive patches more effectively above non-favela patches.")
    lines.append("")
    lines.append("The SAR-only comparison evaluates whether RTC or SNAP-GRD produces more discriminative radar-only CROMA embeddings. The joint comparison evaluates whether the choice of radar preprocessing improves the combined Sentinel-2 plus Sentinel-1 CROMA representation.")
    lines.append("")
    lines.append("The current interpretation should remain Criterion-1-specific. The final preprocessing decision should also consider optical-radar embedding consistency, PCA/UMAP separability, and robustness across cities and regions.")
    lines.append("")

    if output_paths.get("figure_sar_only") is not None:
        lines.append("## Optional generated figures")
        lines.append("")
        lines.append(f"- SAR-only Average Precision figure: `{path_to_str(output_paths['figure_sar_only'])}`")
        lines.append(f"- Joint Average Precision figure: `{path_to_str(output_paths['figure_joint'])}`")
        lines.append("")

    lines.append("## Output files")
    lines.append("")
    for key, value in output_paths.items():
        if value is not None:
            lines.append(f"- `{key}`: `{path_to_str(value)}`")

    lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# Loading fold-specific inputs
# ---------------------------------------------------------------------

def aggregate_path(frozen_probe_dir: Path, fold_mode: str, stem: str) -> Path:
    return frozen_probe_dir / f"frozen_probe_aggregate_{fold_mode}_{stem}.csv"


def comparison_path(frozen_probe_dir: Path, fold_mode: str, stem: str) -> Path:
    return frozen_probe_dir / f"frozen_probe_rtc_vs_snap_comparison_{fold_mode}_{stem}.csv"


def load_fold_outputs(
    frozen_probe_dir: Path,
    fold_modes: Sequence[str],
    stem: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, str]]:
    aggregate_rows: List[Dict[str, str]] = []
    comparison_rows: List[Dict[str, str]] = []
    loaded_files: Dict[str, str] = {}

    for fold_mode in fold_modes:
        agg_path = aggregate_path(frozen_probe_dir, fold_mode, stem)
        comp_path = comparison_path(frozen_probe_dir, fold_mode, stem)

        log("INFO", f"Reading aggregate:  {path_to_str(agg_path)}")
        log("INFO", f"Reading comparison: {path_to_str(comp_path)}")

        agg_rows = read_csv_rows(agg_path)
        comp_rows = read_csv_rows(comp_path)

        for row in agg_rows:
            row["source_fold_mode"] = fold_mode
            aggregate_rows.append(row)

        for row in comp_rows:
            row["source_fold_mode"] = fold_mode
            comparison_rows.append(row)

        loaded_files[f"aggregate_{fold_mode}"] = path_to_str(agg_path)
        loaded_files[f"comparison_{fold_mode}"] = path_to_str(comp_path)

    return aggregate_rows, comparison_rows, loaded_files


# ---------------------------------------------------------------------
# Best modality summary
# ---------------------------------------------------------------------

def build_best_modality_rows(
    aggregate_rows: List[Dict[str, str]],
    primary_metric: str,
) -> List[Dict[str, object]]:
    metric_col = f"{primary_metric}_mean"

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)

    for row in aggregate_rows:
        key = (normalize_fold_mode(row["fold_mode"]), str(row["probe"]))
        grouped[key].append(row)

    best_rows: List[Dict[str, object]] = []

    for (fold_mode, probe), rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda r: safe_float(r.get(metric_col, ""), -1.0),
            reverse=True,
        )

        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else ranked[0]

        best_value = safe_float(best.get(metric_col, ""), 0.0)
        second_value = safe_float(second.get(metric_col, ""), 0.0)

        best_rows.append(
            {
                "fold_mode": fold_mode,
                "fold_mode_pretty": pretty_fold_mode(fold_mode),
                "probe": probe,
                "probe_pretty": pretty_probe(probe),
                "best_modality": best["modality"],
                "best_modality_pretty": pretty_modality(best["modality"]),
                "best_average_precision_mean": round_float(best_value, 8),
                "second_modality": second["modality"],
                "second_modality_pretty": pretty_modality(second["modality"]),
                "second_average_precision_mean": round_float(second_value, 8),
                "margin_best_minus_second": round_float(best_value - second_value, 8),
                "n_folds": safe_int(best.get("n_folds", 0)),
            }
        )

    return best_rows


# ---------------------------------------------------------------------
# RTC-vs-SNAP summary
# ---------------------------------------------------------------------

def direction_from_delta_and_wins(
    delta: float,
    rtc_wins: int,
    snap_wins: int,
    min_mean_delta: float,
) -> str:
    if delta > min_mean_delta and rtc_wins > snap_wins:
        return "RTC stronger"
    if delta < -min_mean_delta and snap_wins > rtc_wins:
        return "SNAP-GRD stronger"
    if abs(delta) <= min_mean_delta:
        if rtc_wins > snap_wins:
            return "Mixed / near tie, RTC more fold wins"
        if snap_wins > rtc_wins:
            return "Mixed / near tie, SNAP-GRD more fold wins"
        return "Near tie"
    if delta > min_mean_delta:
        return "RTC higher mean, mixed folds"
    if delta < -min_mean_delta:
        return "SNAP-GRD higher mean, mixed folds"
    return "Mixed"


def build_metric_comparison_rows(
    comparison_rows: List[Dict[str, str]],
    min_mean_delta: float,
) -> List[Dict[str, object]]:
    out_rows: List[Dict[str, object]] = []

    for row in comparison_rows:
        comparison = row["comparison"]
        fold_mode = normalize_fold_mode(row["fold_mode"])
        probe = row["probe"]
        metric = row["metric"]

        rtc_mean = safe_float(row["rtc_mean"])
        snap_mean = safe_float(row["snap_mean"])
        delta = safe_float(row["delta_rtc_minus_snap"])
        rtc_wins = safe_int(row["rtc_wins"])
        snap_wins = safe_int(row["snap_wins"])
        ties = safe_int(row["ties"])
        n_matched_folds = safe_int(row.get("n_matched_folds", rtc_wins + snap_wins + ties))

        direction = direction_from_delta_and_wins(
            delta=delta,
            rtc_wins=rtc_wins,
            snap_wins=snap_wins,
            min_mean_delta=min_mean_delta,
        )

        out_rows.append(
            {
                "comparison": comparison,
                "comparison_pretty": pretty_comparison(comparison),
                "fold_mode": fold_mode,
                "fold_mode_pretty": pretty_fold_mode(fold_mode),
                "probe": probe,
                "probe_pretty": pretty_probe(probe),
                "metric": metric,
                "n_matched_folds": n_matched_folds,
                "snap_modality": row.get("snap_modality", ""),
                "rtc_modality": row.get("rtc_modality", ""),
                "snap_mean": round_float(snap_mean, 8),
                "rtc_mean": round_float(rtc_mean, 8),
                "delta_rtc_minus_snap": round_float(delta, 8),
                "rtc_wins": rtc_wins,
                "snap_wins": snap_wins,
                "ties": ties,
                "direction": direction,
            }
        )

    return out_rows


def filter_primary_metric_rows(
    all_metric_rows: List[Dict[str, object]],
    primary_metric: str,
) -> List[Dict[str, object]]:
    return [
        row for row in all_metric_rows
        if str(row["metric"]) == primary_metric
    ]


def build_decision_rows(
    primary_rows: List[Dict[str, object]],
    min_mean_delta: float,
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for row in primary_rows:
        grouped[str(row["comparison"])].append(row)

    decision_rows: List[Dict[str, object]] = []

    for comparison, rows in sorted(grouped.items()):
        n_rows = len(rows)

        rtc_higher_mean = sum(
            1 for row in rows
            if safe_float(row["delta_rtc_minus_snap"]) > min_mean_delta
        )

        snap_higher_mean = sum(
            1 for row in rows
            if safe_float(row["delta_rtc_minus_snap"]) < -min_mean_delta
        )

        near_ties = n_rows - rtc_higher_mean - snap_higher_mean

        total_rtc_wins = sum(safe_int(row["rtc_wins"]) for row in rows)
        total_snap_wins = sum(safe_int(row["snap_wins"]) for row in rows)
        total_ties = sum(safe_int(row["ties"]) for row in rows)

        mean_delta = (
            sum(safe_float(row["delta_rtc_minus_snap"]) for row in rows) / n_rows
            if n_rows > 0
            else 0.0
        )

        if comparison == "joint_rtc_vs_snap":
            if snap_higher_mean > rtc_higher_mean and total_snap_wins > total_rtc_wins:
                decision = "Prefer SNAP-GRD"
            elif rtc_higher_mean > snap_higher_mean and total_rtc_wins > total_snap_wins:
                decision = "Prefer RTC"
            else:
                decision = "Mixed / inconclusive"

        elif comparison == "sar_only_rtc_vs_snap":
            if snap_higher_mean > rtc_higher_mean and total_snap_wins > total_rtc_wins:
                decision = "SNAP-GRD stronger overall"
            elif rtc_higher_mean > snap_higher_mean and total_rtc_wins > total_snap_wins:
                decision = "RTC stronger overall"
            else:
                decision = "Very close / mixed"

        else:
            if snap_higher_mean > rtc_higher_mean:
                decision = "SNAP-GRD stronger overall"
            elif rtc_higher_mean > snap_higher_mean:
                decision = "RTC stronger overall"
            else:
                decision = "Mixed / inconclusive"

        evidence_summary = (
            f"Across {n_rows} probe/fold-mode summaries: "
            f"SNAP higher mean AP in {snap_higher_mean}, "
            f"RTC higher mean AP in {rtc_higher_mean}, "
            f"near ties in {near_ties}. "
            f"Fold wins pooled across summaries: "
            f"SNAP={total_snap_wins}, RTC={total_rtc_wins}, ties={total_ties}. "
            f"Mean delta RTC-SNAP={round_float(mean_delta, 8)}."
        )

        decision_rows.append(
            {
                "comparison": comparison,
                "comparison_pretty": pretty_comparison(comparison),
                "decision": decision,
                "n_probe_foldmode_summaries": n_rows,
                "snap_higher_mean_ap_count": snap_higher_mean,
                "rtc_higher_mean_ap_count": rtc_higher_mean,
                "near_tie_count": near_ties,
                "pooled_snap_fold_wins": total_snap_wins,
                "pooled_rtc_fold_wins": total_rtc_wins,
                "pooled_ties": total_ties,
                "mean_delta_rtc_minus_snap": round_float(mean_delta, 8),
                "evidence_summary": evidence_summary,
            }
        )

    return decision_rows


def build_main_conclusion(decision_rows: List[Dict[str, object]]) -> str:
    decision_by_comparison = {
        row["comparison"]: row["decision"]
        for row in decision_rows
    }

    sar_decision = decision_by_comparison.get("sar_only_rtc_vs_snap", "unknown")
    joint_decision = decision_by_comparison.get("joint_rtc_vs_snap", "unknown")

    if "Prefer SNAP-GRD" in joint_decision or "SNAP-GRD" in joint_decision:
        if "Very close" in sar_decision or "mixed" in sar_decision.lower():
            return (
                "Criterion 1 supports SNAP-GRD as the safer current choice. "
                "The SAR-only comparison is close or mixed, but the joint Sentinel-2 plus Sentinel-1 "
                "comparison favours SNAP-GRD. Because the final modelling setup is expected to use "
                "multimodal optical-radar information, the joint result should carry substantial weight."
            )

        if "SNAP-GRD" in sar_decision:
            return (
                "Criterion 1 supports SNAP-GRD. Both the SAR-only and joint Sentinel-2 plus Sentinel-1 "
                "comparisons favour SNAP-GRD overall, indicating that SNAP-GRD currently produces more "
                "useful frozen CROMA representations for this favela patch classification task."
            )

        return (
            "Criterion 1 leans toward SNAP-GRD because the joint Sentinel-2 plus Sentinel-1 comparison "
            "favours SNAP-GRD, even though the SAR-only result is not fully one-sided."
        )

    if "Prefer RTC" in joint_decision or "RTC" in joint_decision:
        if "RTC" in sar_decision:
            return (
                "Criterion 1 supports RTC. Both SAR-only and joint Sentinel-2 plus Sentinel-1 comparisons "
                "favour RTC overall."
            )

        return (
            "Criterion 1 shows evidence in favour of RTC in the joint setting, but the SAR-only result "
            "should be checked carefully before making a final decision."
        )

    return (
        "Criterion 1 is mixed or inconclusive. The RTC versus SNAP-GRD decision should rely on the "
        "remaining criteria: optical-radar embedding consistency, embedding separability, and robustness "
        "analysis."
    )


# ---------------------------------------------------------------------
# Optional figures
# ---------------------------------------------------------------------

def plot_ap_bars(
    primary_rows: List[Dict[str, object]],
    *,
    comparison: str,
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        log("WARN", "matplotlib is not installed; skipping figure generation.")
        return None

    rows = [
        row for row in primary_rows
        if row["comparison"] == comparison
    ]

    if not rows:
        log("WARN", f"No rows found for comparison={comparison}; skipping figure.")
        return None

    ensure_output_can_be_written(output_path, overwrite)

    rows = sorted(rows, key=lambda r: (str(r["fold_mode"]), str(r["probe"])))

    labels = [
        f"{pretty_fold_mode(row['fold_mode'])}\n{pretty_probe(row['probe'])}"
        for row in rows
    ]

    snap_values = [safe_float(row["snap_mean"]) for row in rows]
    rtc_values = [safe_float(row["rtc_mean"]) for row in rows]

    x = list(range(len(rows)))
    width = 0.38

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.bar([i - width / 2 for i in x], snap_values, width, label="SNAP-GRD")
    ax.bar([i + width / 2 for i in x], rtc_values, width, label="RTC")

    ax.set_ylabel("Mean Average Precision")
    ax.set_title(pretty_comparison(comparison))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary_payload(
    *,
    instance_root: Path,
    frozen_probe_dir: Path,
    output_dir: Path,
    fold_modes: Sequence[str],
    primary_metric: str,
    aggregate_rows: List[Dict[str, str]],
    comparison_rows: List[Dict[str, str]],
    best_rows: List[Dict[str, object]],
    all_metric_rows: List[Dict[str, object]],
    primary_rows: List[Dict[str, object]],
    decision_rows: List[Dict[str, object]],
    main_conclusion: str,
    loaded_files: Dict[str, str],
    output_paths: Dict[str, Optional[Path]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "created_utc": now_utc(),
        "status": "passed",
        "instance_root": path_to_str(instance_root),
        "frozen_probe_dir": path_to_str(frozen_probe_dir),
        "output_dir": path_to_str(output_dir),
        "fold_modes": list(fold_modes),
        "primary_metric": primary_metric,
        "n_aggregate_input_rows": len(aggregate_rows),
        "n_comparison_input_rows": len(comparison_rows),
        "n_best_rows": len(best_rows),
        "n_all_metric_rows": len(all_metric_rows),
        "n_primary_rows": len(primary_rows),
        "n_decision_rows": len(decision_rows),
        "main_conclusion": main_conclusion,
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "min_mean_delta": args.min_mean_delta,
            "make_figures": bool(args.make_figures),
        },
        "loaded_files": loaded_files,
        "outputs": {
            key: "" if value is None else path_to_str(value)
            for key, value in output_paths.items()
        },
        "decision_rows": decision_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Criterion 1 frozen-probe outputs for RTC vs SNAP-GRD."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--frozen-probe-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/frozen_probe.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/criterion1_summary.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size. Default: 224.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=112,
        help="Stride. Default: 112.",
    )

    parser.add_argument(
        "--edge-mode",
        choices=["cover", "drop"],
        default="cover",
        help="Edge mode. Default: cover.",
    )

    parser.add_argument(
        "--fold-modes",
        nargs="+",
        default=["leave_one_region", "leave_one_city"],
        choices=["leave_one_region", "leave_one_city"],
        help="Fold modes to summarize.",
    )

    parser.add_argument(
        "--primary-metric",
        default="average_precision",
        help="Primary metric for decision summary. Default: average_precision.",
    )

    parser.add_argument(
        "--min-mean-delta",
        type=float,
        default=1e-4,
        help="Minimum mean delta to treat as non-tie. Default: 1e-4.",
    )

    parser.add_argument(
        "--make-figures",
        action="store_true",
        help="Generate optional Average Precision bar charts if matplotlib is installed.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root

    frozen_probe_dir: Path = args.frozen_probe_dir or (
        instance_root / "metadata" / "croma_probing" / "frozen_probe"
    )

    output_dir: Path = args.output_dir or (
        instance_root / "metadata" / "croma_probing" / "criterion1_summary"
    )

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    best_csv = output_dir / f"criterion1_frozen_probe_best_modalities_{stem}.csv"
    primary_csv = output_dir / f"criterion1_rtc_vs_snap_primary_summary_{stem}.csv"
    all_metrics_csv = output_dir / f"criterion1_rtc_vs_snap_all_metrics_{stem}.csv"
    decision_csv = output_dir / f"criterion1_rtc_vs_snap_decision_{stem}.csv"
    json_path = output_dir / f"criterion1_frozen_probe_summary_{stem}.json"
    md_path = output_dir / f"criterion1_frozen_probe_summary_{stem}.md"

    figure_sar_only: Optional[Path] = None
    figure_joint: Optional[Path] = None

    if args.make_figures:
        figure_dir = output_dir / "figures"
        figure_sar_only = figure_dir / f"criterion1_average_precision_rtc_vs_snap_sar_only_{stem}.png"
        figure_joint = figure_dir / f"criterion1_average_precision_rtc_vs_snap_joint_{stem}.png"

    output_paths: Dict[str, Optional[Path]] = {
        "best_modalities_csv": best_csv,
        "primary_summary_csv": primary_csv,
        "all_metrics_csv": all_metrics_csv,
        "decision_csv": decision_csv,
        "json": json_path,
        "markdown": md_path,
        "figure_sar_only": figure_sar_only,
        "figure_joint": figure_joint,
    }

    log("STEP", "Summarizing Criterion 1 frozen-probe results.")
    log("INFO", f"Instance root:     {path_to_str(instance_root)}")
    log("INFO", f"Frozen-probe dir:  {path_to_str(frozen_probe_dir)}")
    log("INFO", f"Output dir:        {path_to_str(output_dir)}")
    log("INFO", f"Fold modes:        {';'.join(args.fold_modes)}")
    log("INFO", f"Primary metric:    {args.primary_metric}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not frozen_probe_dir.exists():
        fail(f"Frozen-probe directory does not exist: {path_to_str(frozen_probe_dir)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows, comparison_rows, loaded_files = load_fold_outputs(
        frozen_probe_dir=frozen_probe_dir,
        fold_modes=args.fold_modes,
        stem=stem,
    )

    best_rows = build_best_modality_rows(
        aggregate_rows=aggregate_rows,
        primary_metric=str(args.primary_metric),
    )

    all_metric_rows = build_metric_comparison_rows(
        comparison_rows=comparison_rows,
        min_mean_delta=float(args.min_mean_delta),
    )

    primary_rows = filter_primary_metric_rows(
        all_metric_rows=all_metric_rows,
        primary_metric=str(args.primary_metric),
    )

    decision_rows = build_decision_rows(
        primary_rows=primary_rows,
        min_mean_delta=float(args.min_mean_delta),
    )

    main_conclusion = build_main_conclusion(decision_rows)

    if args.make_figures:
        generated_sar = plot_ap_bars(
            primary_rows,
            comparison="sar_only_rtc_vs_snap",
            output_path=figure_sar_only,
            overwrite=bool(args.overwrite),
        )

        generated_joint = plot_ap_bars(
            primary_rows,
            comparison="joint_rtc_vs_snap",
            output_path=figure_joint,
            overwrite=bool(args.overwrite),
        )

        output_paths["figure_sar_only"] = generated_sar
        output_paths["figure_joint"] = generated_joint

    summary_payload = build_summary_payload(
        instance_root=instance_root,
        frozen_probe_dir=frozen_probe_dir,
        output_dir=output_dir,
        fold_modes=args.fold_modes,
        primary_metric=str(args.primary_metric),
        aggregate_rows=aggregate_rows,
        comparison_rows=comparison_rows,
        best_rows=best_rows,
        all_metric_rows=all_metric_rows,
        primary_rows=primary_rows,
        decision_rows=decision_rows,
        main_conclusion=main_conclusion,
        loaded_files=loaded_files,
        output_paths=output_paths,
        args=args,
    )

    log("STEP", "Writing Criterion 1 summary outputs.")

    write_csv(
        best_csv,
        best_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "fold_mode",
            "fold_mode_pretty",
            "probe",
            "probe_pretty",
            "best_modality",
            "best_modality_pretty",
            "best_average_precision_mean",
            "second_modality",
            "second_modality_pretty",
            "second_average_precision_mean",
            "margin_best_minus_second",
            "n_folds",
        ],
    )

    write_csv(
        primary_csv,
        primary_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "comparison",
            "comparison_pretty",
            "fold_mode",
            "fold_mode_pretty",
            "probe",
            "probe_pretty",
            "metric",
            "n_matched_folds",
            "snap_modality",
            "rtc_modality",
            "snap_mean",
            "rtc_mean",
            "delta_rtc_minus_snap",
            "rtc_wins",
            "snap_wins",
            "ties",
            "direction",
        ],
    )

    write_csv(
        all_metrics_csv,
        all_metric_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "comparison",
            "comparison_pretty",
            "fold_mode",
            "fold_mode_pretty",
            "probe",
            "probe_pretty",
            "metric",
            "n_matched_folds",
            "snap_modality",
            "rtc_modality",
            "snap_mean",
            "rtc_mean",
            "delta_rtc_minus_snap",
            "rtc_wins",
            "snap_wins",
            "ties",
            "direction",
        ],
    )

    write_csv(
        decision_csv,
        decision_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "comparison",
            "comparison_pretty",
            "decision",
            "n_probe_foldmode_summaries",
            "snap_higher_mean_ap_count",
            "rtc_higher_mean_ap_count",
            "near_tie_count",
            "pooled_snap_fold_wins",
            "pooled_rtc_fold_wins",
            "pooled_ties",
            "mean_delta_rtc_minus_snap",
            "evidence_summary",
        ],
    )

    write_json(json_path, summary_payload, overwrite=bool(args.overwrite))

    write_markdown(
        md_path,
        summary=summary_payload,
        best_rows=best_rows,
        primary_rows=primary_rows,
        decision_rows=decision_rows,
        output_paths=output_paths,
        overwrite=bool(args.overwrite),
    )

    log("OK", f"Wrote best modalities CSV: {path_to_str(best_csv)}")
    log("OK", f"Wrote primary summary CSV: {path_to_str(primary_csv)}")
    log("OK", f"Wrote all metrics CSV:     {path_to_str(all_metrics_csv)}")
    log("OK", f"Wrote decision CSV:        {path_to_str(decision_csv)}")
    log("OK", f"Wrote JSON:                {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:            {path_to_str(md_path)}")

    if output_paths["figure_sar_only"] is not None:
        log("OK", f"Wrote SAR-only figure:     {path_to_str(output_paths['figure_sar_only'])}")

    if output_paths["figure_joint"] is not None:
        log("OK", f"Wrote joint figure:        {path_to_str(output_paths['figure_joint'])}")

    log("STEP", "Final Criterion 1 summary.")
    log("OK", "Status: passed")
    log("OK", f"Best modality rows: {len(best_rows)}")
    log("OK", f"Primary comparison rows: {len(primary_rows)}")
    log("OK", f"Decision rows: {len(decision_rows)}")
    log("OK", f"Main conclusion: {main_conclusion}")


if __name__ == "__main__":
    main()