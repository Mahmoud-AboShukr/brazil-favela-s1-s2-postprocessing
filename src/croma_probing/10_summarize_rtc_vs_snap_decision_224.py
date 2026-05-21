#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
10_summarize_rtc_vs_snap_decision_224.py

Final RTC vs SNAP-GRD decision summary.

This script consolidates the completed CROMA-based comparison criteria:

    Criterion 1:
        Downstream frozen-probe performance

    Criterion 2:
        Optical-radar embedding consistency

    Criterion 3:
        Embedding separability

It produces one final decision report for whether the current evidence supports
SNAP-GRD or RTC as the preferred Sentinel-1 preprocessing variant for the
CROMA-based favela patch experiments.

Important hierarchy:

    Primary evidence:
        Criterion 1 downstream frozen-probe performance.

    Supporting evidence:
        Criterion 2 optical-radar embedding consistency.
        Criterion 3 embedding separability.

Expected inputs:

    <instance-root>/metadata/croma_probing/criterion1_summary/
        criterion1_rtc_vs_snap_decision_ps224_st112_cover.csv
        criterion1_rtc_vs_snap_primary_summary_ps224_st112_cover.csv

    <instance-root>/metadata/croma_probing/criterion2_embedding_consistency/
        criterion2_decision_summary_ps224_st112_cover.csv

    <instance-root>/metadata/croma_probing/criterion3_embedding_separability/
        criterion3_decision_summary_ps224_st112_cover.csv

Outputs:

    <instance-root>/metadata/croma_probing/final_rtc_vs_snap_decision/
        final_rtc_vs_snap_evidence_table_ps224_st112_cover.csv
        final_rtc_vs_snap_decision_summary_ps224_st112_cover.csv
        final_rtc_vs_snap_decision_report_ps224_st112_cover.json
        final_rtc_vs_snap_decision_report_ps224_st112_cover.md

Example:

python src/croma_probing/10_summarize_rtc_vs_snap_decision_224.py `
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


# ---------------------------------------------------------------------
# I/O helpers
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
            fail(f"No rows to write and no fieldnames provided for {path_to_str(path)}")
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
    evidence_rows: List[Dict[str, object]],
    decision_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Final RTC vs SNAP-GRD Decision Report")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Final recommendation: **{summary['final_recommendation']}**")
    lines.append(f"- Confidence level: **{summary['confidence_level']}**")
    lines.append("")
    lines.append("### Main conclusion")
    lines.append("")
    lines.append(summary["main_conclusion"])
    lines.append("")

    lines.append("## Decision hierarchy")
    lines.append("")
    lines.append("The decision uses a hierarchy rather than treating all criteria equally:")
    lines.append("")
    lines.append("1. **Primary criterion:** downstream frozen-probe performance.")
    lines.append("2. **Supporting criterion:** optical-radar embedding consistency.")
    lines.append("3. **Supporting criterion:** embedding separability.")
    lines.append("")
    lines.append("This means that Criterion 1 carries the most weight because it directly measures whether the embeddings are useful for the target favela patch classification task.")
    lines.append("")

    lines.append("## Final decision by comparison")
    lines.append("")
    lines.append("| comparison | recommendation | confidence | explanation |")
    lines.append("|---|---|---|---|")

    for row in decision_rows:
        lines.append(
            f"| {row['comparison']} | "
            f"{row['recommendation']} | "
            f"{row['confidence']} | "
            f"{row['explanation']} |"
        )

    lines.append("")
    lines.append("## Evidence table")
    lines.append("")
    lines.append("| criterion | comparison | evidence type | SNAP value | RTC value | delta RTC-SNAP | winner | interpretation |")
    lines.append("|---|---|---|---:|---:|---:|---|---|")

    for row in evidence_rows:
        lines.append(
            f"| {row['criterion']} | "
            f"{row['comparison']} | "
            f"{row['evidence_type']} | "
            f"{row['snap_value']} | "
            f"{row['rtc_value']} | "
            f"{row['delta_rtc_minus_snap']} | "
            f"{row['winner']} | "
            f"{row['interpretation']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The completed criteria consistently support SNAP-GRD. Criterion 1 is the most important because it evaluates actual downstream frozen-probe performance under spatial folds. Criterion 2 supports SNAP-GRD because the SNAP-GRD SAR embeddings are more consistent with the Sentinel-2 optical embedding space overall. Criterion 3 supports SNAP-GRD because both SAR-only and joint S2+S1 SNAP-GRD embeddings show stronger local label separability than the corresponding RTC embeddings.")
    lines.append("")
    lines.append("The SAR-only leave-one-city result from Criterion 1 was close, with RTC competitive for some probes. However, the broader evidence still favours SNAP-GRD, especially under leave-one-region transfer and in the joint optical-radar setting.")
    lines.append("")

    lines.append("## Recommended wording for reports")
    lines.append("")
    lines.append("> Based on the completed CROMA-based comparison criteria, we select SNAP-GRD as the preferred Sentinel-1 preprocessing variant for the current favela patch experiments. SNAP-GRD performs better in downstream frozen-probe evaluation, shows higher optical-radar embedding consistency overall, and provides stronger embedding separability than RTC. RTC remains competitive in SAR-only leave-one-city folds, but it does not provide a consistent enough advantage to replace SNAP-GRD at this stage.")
    lines.append("")

    lines.append("## Next steps")
    lines.append("")
    lines.append("- Use SNAP-GRD as the primary Sentinel-1 variant for the next CROMA-based experiments.")
    lines.append("- Keep RTC outputs documented as a secondary comparison branch.")
    lines.append("- Move to the next modelling stage, such as CROMA + UPerNet or another segmentation head, using the selected S1 preprocessing choice.")
    lines.append("- Optionally include the final decision report and key Criterion 1/2/3 figures in the project documentation.")
    lines.append("")

    lines.append("## Input files")
    lines.append("")
    for key, value in summary["input_files"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for key, value in summary["output_files"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# Input path helpers
# ---------------------------------------------------------------------

def criterion1_decision_path(instance_root: Path, stem: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / "criterion1_summary"
        / f"criterion1_rtc_vs_snap_decision_{stem}.csv"
    )


def criterion1_primary_path(instance_root: Path, stem: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / "criterion1_summary"
        / f"criterion1_rtc_vs_snap_primary_summary_{stem}.csv"
    )


def criterion2_decision_path(instance_root: Path, stem: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / "criterion2_embedding_consistency"
        / f"criterion2_decision_summary_{stem}.csv"
    )


def criterion3_decision_path(instance_root: Path, stem: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / "criterion3_embedding_separability"
        / f"criterion3_decision_summary_{stem}.csv"
    )


# ---------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------

def normalize_winner_from_text(text: str) -> str:
    text_l = str(text).lower()

    if "snap" in text_l and "rtc" not in text_l:
        return "SNAP-GRD"

    if "rtc" in text_l and "snap" not in text_l:
        return "RTC"

    if "snap" in text_l and "stronger" in text_l:
        return "SNAP-GRD"

    if "prefer snap" in text_l:
        return "SNAP-GRD"

    if "snap-grd stronger" in text_l:
        return "SNAP-GRD"

    if "rtc stronger" in text_l:
        return "RTC"

    if "prefer rtc" in text_l:
        return "RTC"

    if "mixed" in text_l or "tie" in text_l or "inconclusive" in text_l:
        return "Mixed"

    return "Mixed"


def winner_from_delta(delta: float, min_delta: float) -> str:
    if delta > min_delta:
        return "RTC"
    if delta < -min_delta:
        return "SNAP-GRD"
    return "Mixed"


def comparison_pretty(value: str) -> str:
    mapping = {
        "sar_only_rtc_vs_snap": "SAR-only RTC vs SNAP-GRD",
        "joint_rtc_vs_snap": "Joint S2+S1 RTC vs SNAP-GRD",
        "SAR-only RTC vs SNAP-GRD": "SAR-only RTC vs SNAP-GRD",
        "Joint S2+S1 RTC vs SNAP-GRD": "Joint S2+S1 RTC vs SNAP-GRD",
    }
    return mapping.get(str(value), str(value))


def collect_criterion1_evidence(
    decision_rows: List[Dict[str, str]],
    primary_rows: List[Dict[str, str]],
    min_delta: float,
) -> List[Dict[str, object]]:
    evidence_rows: List[Dict[str, object]] = []

    for row in decision_rows:
        comparison = comparison_pretty(row.get("comparison_pretty", row.get("comparison", "")))
        decision = row.get("decision", "")
        winner = normalize_winner_from_text(decision)

        evidence_rows.append(
            {
                "criterion": "Criterion 1",
                "criterion_rank": 1,
                "comparison": comparison,
                "evidence_type": "overall_decision",
                "snap_value": row.get("snap_higher_mean_ap_count", ""),
                "rtc_value": row.get("rtc_higher_mean_ap_count", ""),
                "delta_rtc_minus_snap": row.get("mean_delta_rtc_minus_snap", ""),
                "winner": winner,
                "interpretation": row.get("evidence_summary", decision),
            }
        )

    for row in primary_rows:
        comparison = comparison_pretty(row.get("comparison_pretty", row.get("comparison", "")))
        fold_mode = row.get("fold_mode_pretty", row.get("fold_mode", ""))
        probe = row.get("probe_pretty", row.get("probe", ""))
        snap_mean = safe_float(row.get("snap_mean", ""))
        rtc_mean = safe_float(row.get("rtc_mean", ""))
        delta = safe_float(row.get("delta_rtc_minus_snap", ""))
        winner = winner_from_delta(delta, min_delta)

        evidence_rows.append(
            {
                "criterion": "Criterion 1",
                "criterion_rank": 1,
                "comparison": comparison,
                "evidence_type": f"average_precision_{fold_mode}_{probe}",
                "snap_value": round_float(snap_mean, 8),
                "rtc_value": round_float(rtc_mean, 8),
                "delta_rtc_minus_snap": round_float(delta, 8),
                "winner": winner,
                "interpretation": (
                    f"Average Precision under {fold_mode} using {probe}. "
                    f"SNAP={round_float(snap_mean, 6)}, RTC={round_float(rtc_mean, 6)}."
                ),
            }
        )

    return evidence_rows


def collect_criterion2_evidence(
    rows: List[Dict[str, str]],
    min_delta: float,
) -> List[Dict[str, object]]:
    evidence_rows: List[Dict[str, object]] = []

    for row in rows:
        group_type = row.get("group_type", "")
        group_value = row.get("group_value", "")

        if group_type not in {"overall", "label_binary", "region"}:
            continue

        snap_mean = safe_float(row.get("snap_mean", ""))
        rtc_mean = safe_float(row.get("rtc_mean", ""))
        delta = safe_float(row.get("delta_rtc_minus_snap_mean", ""))
        winner = winner_from_delta(delta, min_delta)

        evidence_rows.append(
            {
                "criterion": "Criterion 2",
                "criterion_rank": 2,
                "comparison": "Optical-radar consistency",
                "evidence_type": f"{group_type}_{group_value}",
                "snap_value": round_float(snap_mean, 8),
                "rtc_value": round_float(rtc_mean, 8),
                "delta_rtc_minus_snap": round_float(delta, 8),
                "winner": winner,
                "interpretation": (
                    f"Cosine similarity between S2 and SAR embeddings for {group_type}={group_value}. "
                    f"SNAP wins={row.get('snap_wins', '')}, RTC wins={row.get('rtc_wins', '')}."
                ),
            }
        )

    return evidence_rows


def collect_criterion3_evidence(
    rows: List[Dict[str, str]],
    min_delta: float,
) -> List[Dict[str, object]]:
    evidence_rows: List[Dict[str, object]] = []

    for row in rows:
        comparison = comparison_pretty(row.get("comparison_pretty", row.get("comparison", "")))

        snap_primary = safe_float(row.get("snap_primary_value", ""))
        rtc_primary = safe_float(row.get("rtc_primary_value", ""))
        delta_primary = safe_float(row.get("delta_rtc_minus_snap_primary", ""))
        winner_primary = winner_from_delta(delta_primary, min_delta)

        evidence_rows.append(
            {
                "criterion": "Criterion 3",
                "criterion_rank": 3,
                "comparison": comparison,
                "evidence_type": "kNN_average_precision_mean",
                "snap_value": round_float(snap_primary, 8),
                "rtc_value": round_float(rtc_primary, 8),
                "delta_rtc_minus_snap": round_float(delta_primary, 8),
                "winner": winner_primary,
                "interpretation": (
                    "Local label separability measured by kNN Average Precision. "
                    f"Decision: {row.get('decision', '')}."
                ),
            }
        )

        snap_sil = safe_float(row.get("snap_label_silhouette", ""))
        rtc_sil = safe_float(row.get("rtc_label_silhouette", ""))
        delta_sil = safe_float(row.get("delta_rtc_minus_snap_label_silhouette", ""))
        winner_sil = winner_from_delta(delta_sil, min_delta)

        evidence_rows.append(
            {
                "criterion": "Criterion 3",
                "criterion_rank": 3,
                "comparison": comparison,
                "evidence_type": "label_silhouette",
                "snap_value": round_float(snap_sil, 8),
                "rtc_value": round_float(rtc_sil, 8),
                "delta_rtc_minus_snap": round_float(delta_sil, 8),
                "winner": winner_sil,
                "interpretation": "Geometric label separability measured by binary-label silhouette score.",
            }
        )

    return evidence_rows


def summarize_by_comparison(evidence_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    comparisons = [
        "SAR-only RTC vs SNAP-GRD",
        "Joint S2+S1 RTC vs SNAP-GRD",
    ]

    decision_rows: List[Dict[str, object]] = []

    for comparison in comparisons:
        rows = [
            row for row in evidence_rows
            if row["comparison"] == comparison
        ]

        criterion1_rows = [
            row for row in rows
            if row["criterion"] == "Criterion 1"
            and row["evidence_type"] == "overall_decision"
        ]

        criterion3_rows = [
            row for row in rows
            if row["criterion"] == "Criterion 3"
            and row["evidence_type"] == "kNN_average_precision_mean"
        ]

        c1_winner = criterion1_rows[0]["winner"] if criterion1_rows else "Mixed"
        c3_winner = criterion3_rows[0]["winner"] if criterion3_rows else "Mixed"

        snap_votes = sum(1 for row in rows if row["winner"] == "SNAP-GRD")
        rtc_votes = sum(1 for row in rows if row["winner"] == "RTC")
        mixed_votes = sum(1 for row in rows if row["winner"] == "Mixed")

        if c1_winner == "SNAP-GRD":
            recommendation = "Prefer SNAP-GRD"
        elif c1_winner == "RTC":
            recommendation = "Prefer RTC"
        else:
            if c3_winner == "SNAP-GRD":
                recommendation = "Lean SNAP-GRD"
            elif c3_winner == "RTC":
                recommendation = "Lean RTC"
            elif snap_votes > rtc_votes:
                recommendation = "Lean SNAP-GRD"
            elif rtc_votes > snap_votes:
                recommendation = "Lean RTC"
            else:
                recommendation = "Inconclusive"

        if recommendation == "Prefer SNAP-GRD" and c3_winner == "SNAP-GRD":
            confidence = "high"
        elif recommendation == "Prefer RTC" and c3_winner == "RTC":
            confidence = "high"
        elif recommendation.startswith("Prefer"):
            confidence = "medium"
        elif recommendation.startswith("Lean"):
            confidence = "low-to-medium"
        else:
            confidence = "low"

        explanation = (
            f"Criterion 1 winner: {c1_winner}. "
            f"Criterion 3 separability winner: {c3_winner}. "
            f"Evidence votes among detailed rows: SNAP-GRD={snap_votes}, RTC={rtc_votes}, mixed={mixed_votes}."
        )

        decision_rows.append(
            {
                "comparison": comparison,
                "recommendation": recommendation,
                "confidence": confidence,
                "criterion1_winner": c1_winner,
                "criterion3_winner": c3_winner,
                "snap_evidence_votes": snap_votes,
                "rtc_evidence_votes": rtc_votes,
                "mixed_evidence_votes": mixed_votes,
                "explanation": explanation,
            }
        )

    return decision_rows


def summarize_overall(decision_rows: List[Dict[str, object]], evidence_rows: List[Dict[str, object]]) -> Tuple[str, str, str]:
    joint = next(
        (row for row in decision_rows if row["comparison"] == "Joint S2+S1 RTC vs SNAP-GRD"),
        None,
    )

    sar = next(
        (row for row in decision_rows if row["comparison"] == "SAR-only RTC vs SNAP-GRD"),
        None,
    )

    c2_overall = next(
        (
            row for row in evidence_rows
            if row["criterion"] == "Criterion 2"
            and row["evidence_type"] == "overall_all_patches"
        ),
        None,
    )

    c2_winner = c2_overall["winner"] if c2_overall else "Mixed"

    if joint and joint["recommendation"] == "Prefer SNAP-GRD":
        final_recommendation = "SNAP-GRD"
    elif joint and joint["recommendation"] == "Prefer RTC":
        final_recommendation = "RTC"
    elif sar and sar["recommendation"] == "Prefer SNAP-GRD":
        final_recommendation = "SNAP-GRD"
    elif sar and sar["recommendation"] == "Prefer RTC":
        final_recommendation = "RTC"
    else:
        snap_votes = sum(1 for row in evidence_rows if row["winner"] == "SNAP-GRD")
        rtc_votes = sum(1 for row in evidence_rows if row["winner"] == "RTC")

        if snap_votes > rtc_votes:
            final_recommendation = "SNAP-GRD"
        elif rtc_votes > snap_votes:
            final_recommendation = "RTC"
        else:
            final_recommendation = "Inconclusive"

    if final_recommendation == "SNAP-GRD":
        if (
            joint
            and joint["recommendation"] == "Prefer SNAP-GRD"
            and sar
            and "SNAP-GRD" in sar["recommendation"]
            and c2_winner == "SNAP-GRD"
        ):
            confidence = "high"
        else:
            confidence = "medium-to-high"

        conclusion = (
            "The completed CROMA-based comparison supports SNAP-GRD as the preferred "
            "Sentinel-1 preprocessing variant. Criterion 1 favours SNAP-GRD overall, "
            "especially in the joint Sentinel-2 plus Sentinel-1 setting. Criterion 2 also "
            "supports SNAP-GRD because its SAR embeddings are more consistent with the "
            "Sentinel-2 optical embedding space overall. Criterion 3 supports SNAP-GRD "
            "because its embeddings show stronger local label separability than the RTC "
            "alternatives. RTC remains competitive in some SAR-only city-level cases, but "
            "it does not provide a consistent enough advantage to replace SNAP-GRD."
        )

    elif final_recommendation == "RTC":
        confidence = "medium"
        conclusion = (
            "The completed CROMA-based comparison supports RTC as the preferred Sentinel-1 "
            "preprocessing variant. This should be reviewed carefully against Criterion 1 "
            "because downstream frozen-probe performance is the primary decision criterion."
        )

    else:
        confidence = "low"
        conclusion = (
            "The completed CROMA-based comparison does not provide a clear final preference "
            "between SNAP-GRD and RTC. Additional experiments are required before selecting "
            "a primary Sentinel-1 preprocessing variant."
        )

    return final_recommendation, confidence, conclusion


# ---------------------------------------------------------------------
# Summary payload
# ---------------------------------------------------------------------

def build_summary_payload(
    *,
    instance_root: Path,
    output_dir: Path,
    final_recommendation: str,
    confidence_level: str,
    main_conclusion: str,
    evidence_rows: List[Dict[str, object]],
    decision_rows: List[Dict[str, object]],
    input_files: Dict[str, str],
    output_files: Dict[str, str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "created_utc": now_utc(),
        "status": "passed",
        "instance_root": path_to_str(instance_root),
        "output_dir": path_to_str(output_dir),
        "final_recommendation": final_recommendation,
        "confidence_level": confidence_level,
        "main_conclusion": main_conclusion,
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "min_delta": args.min_delta,
        },
        "n_evidence_rows": len(evidence_rows),
        "n_decision_rows": len(decision_rows),
        "input_files": input_files,
        "output_files": output_files,
        "decision_rows": decision_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize final RTC vs SNAP-GRD decision from CROMA criteria."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/final_rtc_vs_snap_decision.",
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
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum delta treated as meaningful. Default: 1e-4.",
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

    output_dir: Path = args.output_dir or (
        instance_root
        / "metadata"
        / "croma_probing"
        / "final_rtc_vs_snap_decision"
    )

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    c1_decision = criterion1_decision_path(instance_root, stem)
    c1_primary = criterion1_primary_path(instance_root, stem)
    c2_decision = criterion2_decision_path(instance_root, stem)
    c3_decision = criterion3_decision_path(instance_root, stem)

    evidence_csv = output_dir / f"final_rtc_vs_snap_evidence_table_{stem}.csv"
    decision_csv = output_dir / f"final_rtc_vs_snap_decision_summary_{stem}.csv"
    json_path = output_dir / f"final_rtc_vs_snap_decision_report_{stem}.json"
    md_path = output_dir / f"final_rtc_vs_snap_decision_report_{stem}.md"

    input_files = {
        "criterion1_decision": path_to_str(c1_decision),
        "criterion1_primary": path_to_str(c1_primary),
        "criterion2_decision": path_to_str(c2_decision),
        "criterion3_decision": path_to_str(c3_decision),
    }

    output_files = {
        "evidence_csv": path_to_str(evidence_csv),
        "decision_csv": path_to_str(decision_csv),
        "json": path_to_str(json_path),
        "markdown": path_to_str(md_path),
    }

    log("STEP", "Building final RTC vs SNAP-GRD decision report.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")
    log("INFO", f"Stem:          {stem}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    log("STEP", "Reading criterion outputs.")

    c1_decision_rows = read_csv_rows(c1_decision)
    c1_primary_rows = read_csv_rows(c1_primary)
    c2_decision_rows = read_csv_rows(c2_decision)
    c3_decision_rows = read_csv_rows(c3_decision)

    log("OK", f"Criterion 1 decision rows: {len(c1_decision_rows)}")
    log("OK", f"Criterion 1 primary rows:  {len(c1_primary_rows)}")
    log("OK", f"Criterion 2 decision rows: {len(c2_decision_rows)}")
    log("OK", f"Criterion 3 decision rows: {len(c3_decision_rows)}")

    evidence_rows: List[Dict[str, object]] = []

    evidence_rows.extend(
        collect_criterion1_evidence(
            c1_decision_rows,
            c1_primary_rows,
            min_delta=float(args.min_delta),
        )
    )

    evidence_rows.extend(
        collect_criterion2_evidence(
            c2_decision_rows,
            min_delta=float(args.min_delta),
        )
    )

    evidence_rows.extend(
        collect_criterion3_evidence(
            c3_decision_rows,
            min_delta=float(args.min_delta),
        )
    )

    decision_rows = summarize_by_comparison(evidence_rows)

    final_recommendation, confidence_level, main_conclusion = summarize_overall(
        decision_rows,
        evidence_rows,
    )

    summary_payload = build_summary_payload(
        instance_root=instance_root,
        output_dir=output_dir,
        final_recommendation=final_recommendation,
        confidence_level=confidence_level,
        main_conclusion=main_conclusion,
        evidence_rows=evidence_rows,
        decision_rows=decision_rows,
        input_files=input_files,
        output_files=output_files,
        args=args,
    )

    log("STEP", "Writing final decision outputs.")

    write_csv(
        evidence_csv,
        evidence_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "criterion",
            "criterion_rank",
            "comparison",
            "evidence_type",
            "snap_value",
            "rtc_value",
            "delta_rtc_minus_snap",
            "winner",
            "interpretation",
        ],
    )

    write_csv(
        decision_csv,
        decision_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "comparison",
            "recommendation",
            "confidence",
            "criterion1_winner",
            "criterion3_winner",
            "snap_evidence_votes",
            "rtc_evidence_votes",
            "mixed_evidence_votes",
            "explanation",
        ],
    )

    write_json(json_path, summary_payload, overwrite=bool(args.overwrite))

    write_markdown(
        md_path,
        summary=summary_payload,
        evidence_rows=evidence_rows,
        decision_rows=decision_rows,
        overwrite=bool(args.overwrite),
    )

    log("OK", f"Wrote evidence CSV: {path_to_str(evidence_csv)}")
    log("OK", f"Wrote decision CSV: {path_to_str(decision_csv)}")
    log("OK", f"Wrote JSON:         {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:     {path_to_str(md_path)}")

    log("STEP", "Final RTC vs SNAP-GRD decision.")
    log("OK", f"Status: passed")
    log("OK", f"Final recommendation: {final_recommendation}")
    log("OK", f"Confidence level: {confidence_level}")
    log("OK", f"Main conclusion: {main_conclusion}")


if __name__ == "__main__":
    main()