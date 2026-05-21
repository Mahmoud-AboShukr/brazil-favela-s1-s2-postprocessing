#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
08_compute_optical_radar_embedding_consistency_224.py

Criterion 2:
    Optical-radar embedding consistency.

This script compares how close the SAR-only CROMA embeddings are to the S2-only
CROMA embeddings.

It computes, for every patch:

    cosine(S2 optical embedding, S1 SNAP-GRD SAR embedding)
    cosine(S2 optical embedding, S1 RTC SAR embedding)

Then it compares:

    SNAP-GRD cosine similarity
    RTC cosine similarity
    delta = RTC - SNAP-GRD

Interpretation:
    Higher and more stable cosine similarity suggests that the SAR embedding is
    more consistent with the optical representation space.

Important:
    This is supporting evidence only.
    We should not select RTC or SNAP-GRD based on this criterion alone.

Expected inputs:

    <instance-root>/metadata/croma_probing/full_embeddings/
        croma_embeddings_s2_ps224_st112_cover.npz
        croma_embeddings_s1_snap_vv_vh_ps224_st112_cover.npz
        croma_embeddings_s1_rtc_vv_vh_ps224_st112_cover.npz

Outputs:

    <instance-root>/metadata/croma_probing/criterion2_embedding_consistency/
        criterion2_patchwise_cosine_ps224_st112_cover.csv
        criterion2_group_summary_ps224_st112_cover.csv
        criterion2_decision_summary_ps224_st112_cover.csv
        criterion2_embedding_consistency_summary_ps224_st112_cover.json
        criterion2_embedding_consistency_summary_ps224_st112_cover.md

Optional figures if --make-figures is used:
        figures/criterion2_cosine_histogram_ps224_st112_cover.png
        figures/criterion2_region_mean_cosine_ps224_st112_cover.png
        figures/criterion2_city_delta_rtc_minus_snap_ps224_st112_cover.png

Example:

python src/croma_probing/08_compute_optical_radar_embedding_consistency_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --make-figures `
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

import numpy as np


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


def pretty_label(value: int) -> str:
    if int(value) == 1:
        return "positive"
    return "empty"


# ---------------------------------------------------------------------
# CSV / JSON / Markdown
# ---------------------------------------------------------------------

def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    overwrite: bool,
    fieldnames: Optional[List[str]] = None,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    if fieldnames is None:
        if not rows:
            fail(f"No rows to write and no fieldnames were provided: {path_to_str(path)}")
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
    group_rows: List[Dict[str, object]],
    decision_rows: List[Dict[str, object]],
    output_paths: Dict[str, Optional[Path]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Criterion 2 Summary: Optical-Radar Embedding Consistency")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Embedding directory: `{summary['embedding_dir']}`")
    lines.append(f"- Output directory: `{summary['output_dir']}`")
    lines.append(f"- Number of patches: `{summary['n_patches']}`")
    lines.append(f"- Embedding dimension: `{summary['embedding_dim']}`")
    lines.append("")

    lines.append("### Main conclusion")
    lines.append("")
    lines.append(summary["main_conclusion"])
    lines.append("")

    lines.append("## Decision summary")
    lines.append("")
    lines.append("| level | group | n | SNAP mean cosine | RTC mean cosine | delta RTC-SNAP | RTC wins | SNAP wins | decision |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")

    for row in decision_rows:
        lines.append(
            f"| {row['group_type']} | "
            f"{row['group_value']} | "
            f"{row['n']} | "
            f"{row['snap_mean']} | "
            f"{row['rtc_mean']} | "
            f"{row['delta_rtc_minus_snap_mean']} | "
            f"{row['rtc_wins']} | "
            f"{row['snap_wins']} | "
            f"{row['decision']} |"
        )

    lines.append("")
    lines.append("## Group-level summaries")
    lines.append("")
    lines.append("| group type | group value | n | SNAP mean | SNAP std | RTC mean | RTC std | delta mean | RTC win % | SNAP win % |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in group_rows:
        if row["group_type"] not in {"overall", "label_binary", "region"}:
            continue

        lines.append(
            f"| {row['group_type']} | "
            f"{row['group_value']} | "
            f"{row['n']} | "
            f"{row['snap_mean']} | "
            f"{row['snap_std']} | "
            f"{row['rtc_mean']} | "
            f"{row['rtc_std']} | "
            f"{row['delta_rtc_minus_snap_mean']} | "
            f"{row['rtc_win_percent']} | "
            f"{row['snap_win_percent']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This criterion measures the cosine similarity between the optical CROMA embedding and each SAR-only CROMA embedding for the same patch. A higher cosine similarity means that the SAR representation is more aligned with the optical representation space.")
    lines.append("")
    lines.append("This should be treated as supporting evidence only. A higher optical-radar similarity does not automatically mean better downstream task performance. The primary selection criterion remains the frozen-probe performance from Criterion 1.")
    lines.append("")

    if output_paths.get("figure_histogram") is not None:
        lines.append("## Optional generated figures")
        lines.append("")
        lines.append(f"- Cosine histogram: `{path_to_str(output_paths['figure_histogram'])}`")
        lines.append(f"- Region mean cosine: `{path_to_str(output_paths['figure_region'])}`")
        lines.append(f"- City delta RTC-SNAP: `{path_to_str(output_paths['figure_city_delta'])}`")
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
# Embedding loading
# ---------------------------------------------------------------------

def embedding_path_for_modality(
    embedding_dir: Path,
    modality: str,
    stem: str,
) -> Path:
    return embedding_dir / f"croma_embeddings_{modality}_{stem}.npz"


def load_embedding_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        fail(f"Embedding file does not exist: {path_to_str(path)}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_required_embeddings(
    embedding_dir: Path,
    stem: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    modalities = [
        "s2",
        "s1_snap_vv_vh",
        "s1_rtc_vv_vh",
    ]

    loaded: Dict[str, Dict[str, np.ndarray]] = {}

    for modality in modalities:
        path = embedding_path_for_modality(embedding_dir, modality, stem)
        arrays = load_embedding_npz(path)
        loaded[modality] = arrays

        emb = arrays["embeddings"]

        log(
            "OK",
            f"{modality}: loaded {path_to_str(path)} | shape={emb.shape}",
        )

    return loaded


def validate_alignment(loaded: Dict[str, Dict[str, np.ndarray]]) -> None:
    reference = loaded["s2"]

    for modality in ["s1_snap_vv_vh", "s1_rtc_vv_vh"]:
        arrays = loaded[modality]

        for key in [
            "patch_ids",
            "label_binary",
            "cities",
            "regions",
            "label_positive_pixels",
            "label_density_bins",
        ]:
            if not np.array_equal(reference[key], arrays[key]):
                fail(f"Metadata alignment failed for modality={modality}, key={key}")

        ref_percent = reference["label_positive_percent"].astype(np.float32)
        cur_percent = arrays["label_positive_percent"].astype(np.float32)

        if not np.allclose(ref_percent, cur_percent, atol=1e-6, rtol=0.0):
            fail(f"label_positive_percent alignment failed for modality={modality}")

    s2_shape = reference["embeddings"].shape
    snap_shape = loaded["s1_snap_vv_vh"]["embeddings"].shape
    rtc_shape = loaded["s1_rtc_vv_vh"]["embeddings"].shape

    if s2_shape != snap_shape:
        fail(f"S2 and SNAP embedding shapes differ: {s2_shape} vs {snap_shape}")

    if s2_shape != rtc_shape:
        fail(f"S2 and RTC embedding shapes differ: {s2_shape} vs {rtc_shape}")

    if len(s2_shape) != 2:
        fail(f"Expected 2D embeddings, got shape={s2_shape}")

    log("OK", "S2, SNAP-GRD, and RTC embeddings are aligned.")


# ---------------------------------------------------------------------
# Cosine computation
# ---------------------------------------------------------------------

def rowwise_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    numerator = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)

    cosine = numerator / np.maximum(denom, eps)

    return cosine.astype(np.float32)


def winner_from_delta(delta: float, tolerance: float) -> str:
    if delta > tolerance:
        return "rtc"
    if delta < -tolerance:
        return "snap"
    return "tie"


# ---------------------------------------------------------------------
# Patchwise rows and summaries
# ---------------------------------------------------------------------

def build_patchwise_rows(
    *,
    metadata: Dict[str, np.ndarray],
    cos_snap: np.ndarray,
    cos_rtc: np.ndarray,
    tolerance: float,
) -> List[Dict[str, object]]:
    patch_ids = metadata["patch_ids"].astype(str)
    cities = metadata["cities"].astype(str)
    regions = metadata["regions"].astype(str)
    labels = metadata["label_binary"].astype(np.int64)
    label_positive_pixels = metadata["label_positive_pixels"].astype(np.int64)
    label_positive_percent = metadata["label_positive_percent"].astype(np.float32)
    density_bins = metadata["label_density_bins"].astype(str)

    rows: List[Dict[str, object]] = []

    for i in range(len(patch_ids)):
        delta = float(cos_rtc[i] - cos_snap[i])
        winner = winner_from_delta(delta, tolerance)

        rows.append(
            {
                "patch_index": i,
                "patch_id": patch_ids[i],
                "city": cities[i],
                "region": regions[i],
                "label_binary": int(labels[i]),
                "label_name": pretty_label(int(labels[i])),
                "label_positive_pixels": int(label_positive_pixels[i]),
                "label_positive_percent": round_float(float(label_positive_percent[i]), 8),
                "label_density_bin": density_bins[i],
                "cosine_s2_snap": round_float(float(cos_snap[i]), 10),
                "cosine_s2_rtc": round_float(float(cos_rtc[i]), 10),
                "delta_rtc_minus_snap": round_float(delta, 10),
                "higher_similarity_variant": winner,
            }
        )

    return rows


def summarize_values(
    values: np.ndarray,
) -> Dict[str, float]:
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "mean": round_float(float(np.mean(values)), 10),
        "std": round_float(float(np.std(values)), 10),
        "median": round_float(float(np.median(values)), 10),
        "q25": round_float(float(np.quantile(values, 0.25)), 10),
        "q75": round_float(float(np.quantile(values, 0.75)), 10),
        "min": round_float(float(np.min(values)), 10),
        "max": round_float(float(np.max(values)), 10),
    }


def summarize_group(
    *,
    group_type: str,
    group_value: str,
    mask: np.ndarray,
    labels: np.ndarray,
    cos_snap: np.ndarray,
    cos_rtc: np.ndarray,
    tolerance: float,
) -> Dict[str, object]:
    snap_vals = cos_snap[mask]
    rtc_vals = cos_rtc[mask]
    delta_vals = rtc_vals - snap_vals

    snap_stats = summarize_values(snap_vals)
    rtc_stats = summarize_values(rtc_vals)
    delta_stats = summarize_values(delta_vals)

    n = int(np.count_nonzero(mask))
    positive = int(np.count_nonzero(labels[mask] == 1))
    empty = int(np.count_nonzero(labels[mask] == 0))

    rtc_wins = int(np.count_nonzero(delta_vals > tolerance))
    snap_wins = int(np.count_nonzero(delta_vals < -tolerance))
    ties = int(n - rtc_wins - snap_wins)

    if n > 0:
        rtc_win_percent = 100.0 * rtc_wins / n
        snap_win_percent = 100.0 * snap_wins / n
        tie_percent = 100.0 * ties / n
    else:
        rtc_win_percent = 0.0
        snap_win_percent = 0.0
        tie_percent = 0.0

    if delta_stats["mean"] > tolerance and rtc_wins > snap_wins:
        decision = "RTC higher consistency"
    elif delta_stats["mean"] < -tolerance and snap_wins > rtc_wins:
        decision = "SNAP-GRD higher consistency"
    elif abs(delta_stats["mean"]) <= tolerance:
        decision = "Near tie"
    elif delta_stats["mean"] > tolerance:
        decision = "RTC higher mean, mixed patches"
    else:
        decision = "SNAP-GRD higher mean, mixed patches"

    return {
        "group_type": group_type,
        "group_value": group_value,
        "n": n,
        "positive_count": positive,
        "empty_count": empty,

        "snap_mean": snap_stats["mean"],
        "snap_std": snap_stats["std"],
        "snap_median": snap_stats["median"],
        "snap_q25": snap_stats["q25"],
        "snap_q75": snap_stats["q75"],
        "snap_min": snap_stats["min"],
        "snap_max": snap_stats["max"],

        "rtc_mean": rtc_stats["mean"],
        "rtc_std": rtc_stats["std"],
        "rtc_median": rtc_stats["median"],
        "rtc_q25": rtc_stats["q25"],
        "rtc_q75": rtc_stats["q75"],
        "rtc_min": rtc_stats["min"],
        "rtc_max": rtc_stats["max"],

        "delta_rtc_minus_snap_mean": delta_stats["mean"],
        "delta_rtc_minus_snap_std": delta_stats["std"],
        "delta_rtc_minus_snap_median": delta_stats["median"],
        "delta_rtc_minus_snap_q25": delta_stats["q25"],
        "delta_rtc_minus_snap_q75": delta_stats["q75"],
        "delta_rtc_minus_snap_min": delta_stats["min"],
        "delta_rtc_minus_snap_max": delta_stats["max"],

        "rtc_wins": rtc_wins,
        "snap_wins": snap_wins,
        "ties": ties,
        "rtc_win_percent": round_float(rtc_win_percent, 6),
        "snap_win_percent": round_float(snap_win_percent, 6),
        "tie_percent": round_float(tie_percent, 6),
        "decision": decision,
    }


def build_group_summary_rows(
    *,
    metadata: Dict[str, np.ndarray],
    cos_snap: np.ndarray,
    cos_rtc: np.ndarray,
    tolerance: float,
) -> List[Dict[str, object]]:
    labels = metadata["label_binary"].astype(np.int64)
    cities = metadata["cities"].astype(str)
    regions = metadata["regions"].astype(str)
    density_bins = metadata["label_density_bins"].astype(str)

    n = len(labels)

    rows: List[Dict[str, object]] = []

    rows.append(
        summarize_group(
            group_type="overall",
            group_value="all_patches",
            mask=np.ones(n, dtype=bool),
            labels=labels,
            cos_snap=cos_snap,
            cos_rtc=cos_rtc,
            tolerance=tolerance,
        )
    )

    for value in sorted(set(labels.tolist())):
        rows.append(
            summarize_group(
                group_type="label_binary",
                group_value=f"{int(value)}_{pretty_label(int(value))}",
                mask=labels == int(value),
                labels=labels,
                cos_snap=cos_snap,
                cos_rtc=cos_rtc,
                tolerance=tolerance,
            )
        )

    for value in sorted(set(regions.tolist())):
        rows.append(
            summarize_group(
                group_type="region",
                group_value=str(value),
                mask=regions == str(value),
                labels=labels,
                cos_snap=cos_snap,
                cos_rtc=cos_rtc,
                tolerance=tolerance,
            )
        )

    for value in sorted(set(density_bins.tolist())):
        rows.append(
            summarize_group(
                group_type="label_density_bin",
                group_value=str(value),
                mask=density_bins == str(value),
                labels=labels,
                cos_snap=cos_snap,
                cos_rtc=cos_rtc,
                tolerance=tolerance,
            )
        )

    for value in sorted(set(cities.tolist())):
        rows.append(
            summarize_group(
                group_type="city",
                group_value=str(value),
                mask=cities == str(value),
                labels=labels,
                cos_snap=cos_snap,
                cos_rtc=cos_rtc,
                tolerance=tolerance,
            )
        )

    return rows


def build_decision_rows(group_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    decision_rows: List[Dict[str, object]] = []

    for row in group_rows:
        if row["group_type"] not in {"overall", "label_binary", "region"}:
            continue

        decision_rows.append(
            {
                "group_type": row["group_type"],
                "group_value": row["group_value"],
                "n": row["n"],
                "snap_mean": row["snap_mean"],
                "rtc_mean": row["rtc_mean"],
                "delta_rtc_minus_snap_mean": row["delta_rtc_minus_snap_mean"],
                "rtc_wins": row["rtc_wins"],
                "snap_wins": row["snap_wins"],
                "ties": row["ties"],
                "rtc_win_percent": row["rtc_win_percent"],
                "snap_win_percent": row["snap_win_percent"],
                "decision": row["decision"],
            }
        )

    return decision_rows


def build_main_conclusion(decision_rows: List[Dict[str, object]]) -> str:
    overall = None

    for row in decision_rows:
        if row["group_type"] == "overall":
            overall = row
            break

    if overall is None:
        return "Criterion 2 could not produce an overall decision row."

    delta = safe_float(overall["delta_rtc_minus_snap_mean"])
    decision = str(overall["decision"])
    snap_mean = safe_float(overall["snap_mean"])
    rtc_mean = safe_float(overall["rtc_mean"])
    snap_wins = safe_int(overall["snap_wins"])
    rtc_wins = safe_int(overall["rtc_wins"])

    if "SNAP-GRD" in decision:
        return (
            "Criterion 2 supports SNAP-GRD as the SAR variant with higher optical-radar "
            "embedding consistency. Overall, SNAP-GRD has higher mean cosine similarity "
            f"with S2 than RTC (SNAP={round_float(snap_mean, 6)}, "
            f"RTC={round_float(rtc_mean, 6)}, delta RTC-SNAP={round_float(delta, 6)}), "
            f"and SNAP-GRD wins more patches ({snap_wins} versus {rtc_wins}). "
            "This is supporting evidence and should be interpreted together with Criterion 1."
        )

    if "RTC" in decision:
        return (
            "Criterion 2 supports RTC as the SAR variant with higher optical-radar "
            "embedding consistency. Overall, RTC has higher mean cosine similarity "
            f"with S2 than SNAP-GRD (RTC={round_float(rtc_mean, 6)}, "
            f"SNAP={round_float(snap_mean, 6)}, delta RTC-SNAP={round_float(delta, 6)}), "
            f"and RTC wins more patches ({rtc_wins} versus {snap_wins}). "
            "This is supporting evidence and should be interpreted together with Criterion 1."
        )

    return (
        "Criterion 2 is close or mixed. The overall optical-radar cosine similarity does "
        "not provide a strong standalone preference between SNAP-GRD and RTC. This criterion "
        "should be treated only as supporting evidence."
    )


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def make_histogram_figure(
    *,
    cos_snap: np.ndarray,
    cos_rtc: np.ndarray,
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        log("WARN", "matplotlib is not installed; skipping histogram figure.")
        return None

    ensure_output_can_be_written(output_path, overwrite)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.hist(cos_snap, bins=60, alpha=0.6, label="S2 vs SNAP-GRD")
    ax.hist(cos_rtc, bins=60, alpha=0.6, label="S2 vs RTC")

    ax.set_title("Optical-radar embedding consistency")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Number of patches")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def make_region_bar_figure(
    *,
    group_rows: List[Dict[str, object]],
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        log("WARN", "matplotlib is not installed; skipping region figure.")
        return None

    region_rows = [
        row for row in group_rows
        if row["group_type"] == "region"
    ]

    if not region_rows:
        return None

    region_rows = sorted(region_rows, key=lambda r: str(r["group_value"]))

    ensure_output_can_be_written(output_path, overwrite)

    labels = [str(row["group_value"]) for row in region_rows]
    snap_values = [safe_float(row["snap_mean"]) for row in region_rows]
    rtc_values = [safe_float(row["rtc_mean"]) for row in region_rows]

    x = list(range(len(region_rows)))
    width = 0.38

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.bar([i - width / 2 for i in x], snap_values, width, label="SNAP-GRD")
    ax.bar([i + width / 2 for i in x], rtc_values, width, label="RTC")

    ax.set_title("Mean optical-radar cosine similarity by region")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def make_city_delta_figure(
    *,
    group_rows: List[Dict[str, object]],
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        log("WARN", "matplotlib is not installed; skipping city delta figure.")
        return None

    city_rows = [
        row for row in group_rows
        if row["group_type"] == "city"
    ]

    if not city_rows:
        return None

    city_rows = sorted(
        city_rows,
        key=lambda r: safe_float(r["delta_rtc_minus_snap_mean"]),
    )

    ensure_output_can_be_written(output_path, overwrite)

    labels = [str(row["group_value"]) for row in city_rows]
    values = [safe_float(row["delta_rtc_minus_snap_mean"]) for row in city_rows]

    fig = plt.figure(figsize=(10, max(6, 0.25 * len(city_rows))))
    ax = fig.add_subplot(111)

    y = list(range(len(city_rows)))

    ax.barh(y, values)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean delta cosine similarity: RTC - SNAP-GRD")
    ax.set_title("City-level optical-radar consistency difference")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------
# Summary payload
# ---------------------------------------------------------------------

def build_summary_payload(
    *,
    instance_root: Path,
    embedding_dir: Path,
    output_dir: Path,
    n_patches: int,
    embedding_dim: int,
    group_rows: List[Dict[str, object]],
    decision_rows: List[Dict[str, object]],
    main_conclusion: str,
    output_paths: Dict[str, Optional[Path]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "created_utc": now_utc(),
        "status": "passed",
        "instance_root": path_to_str(instance_root),
        "embedding_dir": path_to_str(embedding_dir),
        "output_dir": path_to_str(output_dir),
        "n_patches": int(n_patches),
        "embedding_dim": int(embedding_dim),
        "main_conclusion": main_conclusion,
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "cosine_tolerance": args.cosine_tolerance,
            "make_figures": bool(args.make_figures),
        },
        "outputs": {
            key: "" if value is None else path_to_str(value)
            for key, value in output_paths.items()
        },
        "decision_rows": decision_rows,
        "overall_row": [
            row for row in group_rows
            if row["group_type"] == "overall"
        ],
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute optical-radar embedding consistency for CROMA embeddings."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/full_embeddings.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/criterion2_embedding_consistency.",
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
        "--cosine-tolerance",
        type=float,
        default=1e-8,
        help="Tolerance for declaring a patch-level cosine tie. Default: 1e-8.",
    )

    parser.add_argument(
        "--make-figures",
        action="store_true",
        help="Generate optional figures if matplotlib is available.",
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

    embedding_dir: Path = args.embedding_dir or (
        instance_root / "metadata" / "croma_probing" / "full_embeddings"
    )

    output_dir: Path = args.output_dir or (
        instance_root / "metadata" / "croma_probing" / "criterion2_embedding_consistency"
    )

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    patchwise_csv = output_dir / f"criterion2_patchwise_cosine_{stem}.csv"
    group_summary_csv = output_dir / f"criterion2_group_summary_{stem}.csv"
    decision_csv = output_dir / f"criterion2_decision_summary_{stem}.csv"
    json_path = output_dir / f"criterion2_embedding_consistency_summary_{stem}.json"
    md_path = output_dir / f"criterion2_embedding_consistency_summary_{stem}.md"

    figure_histogram: Optional[Path] = None
    figure_region: Optional[Path] = None
    figure_city_delta: Optional[Path] = None

    if args.make_figures:
        figure_dir = output_dir / "figures"
        figure_histogram = figure_dir / f"criterion2_cosine_histogram_{stem}.png"
        figure_region = figure_dir / f"criterion2_region_mean_cosine_{stem}.png"
        figure_city_delta = figure_dir / f"criterion2_city_delta_rtc_minus_snap_{stem}.png"

    output_paths: Dict[str, Optional[Path]] = {
        "patchwise_csv": patchwise_csv,
        "group_summary_csv": group_summary_csv,
        "decision_csv": decision_csv,
        "json": json_path,
        "markdown": md_path,
        "figure_histogram": figure_histogram,
        "figure_region": figure_region,
        "figure_city_delta": figure_city_delta,
    }

    log("STEP", "Computing Criterion 2 optical-radar embedding consistency.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Embedding dir: {path_to_str(embedding_dir)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not embedding_dir.exists():
        fail(f"Embedding directory does not exist: {path_to_str(embedding_dir)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_required_embeddings(
        embedding_dir=embedding_dir,
        stem=stem,
    )

    validate_alignment(loaded)

    s2_embeddings = loaded["s2"]["embeddings"].astype(np.float32)
    snap_embeddings = loaded["s1_snap_vv_vh"]["embeddings"].astype(np.float32)
    rtc_embeddings = loaded["s1_rtc_vv_vh"]["embeddings"].astype(np.float32)

    metadata = loaded["s2"]

    n_patches = int(s2_embeddings.shape[0])
    embedding_dim = int(s2_embeddings.shape[1])

    log("STEP", "Computing row-wise cosine similarities.")

    cos_snap = rowwise_cosine(s2_embeddings, snap_embeddings)
    cos_rtc = rowwise_cosine(s2_embeddings, rtc_embeddings)

    if not np.isfinite(cos_snap).all():
        fail("Non-finite values found in S2-SNAP cosine similarities.")

    if not np.isfinite(cos_rtc).all():
        fail("Non-finite values found in S2-RTC cosine similarities.")

    patchwise_rows = build_patchwise_rows(
        metadata=metadata,
        cos_snap=cos_snap,
        cos_rtc=cos_rtc,
        tolerance=float(args.cosine_tolerance),
    )

    group_rows = build_group_summary_rows(
        metadata=metadata,
        cos_snap=cos_snap,
        cos_rtc=cos_rtc,
        tolerance=float(args.cosine_tolerance),
    )

    decision_rows = build_decision_rows(group_rows)

    main_conclusion = build_main_conclusion(decision_rows)

    if args.make_figures:
        generated_hist = make_histogram_figure(
            cos_snap=cos_snap,
            cos_rtc=cos_rtc,
            output_path=figure_histogram,
            overwrite=bool(args.overwrite),
        )

        generated_region = make_region_bar_figure(
            group_rows=group_rows,
            output_path=figure_region,
            overwrite=bool(args.overwrite),
        )

        generated_city = make_city_delta_figure(
            group_rows=group_rows,
            output_path=figure_city_delta,
            overwrite=bool(args.overwrite),
        )

        output_paths["figure_histogram"] = generated_hist
        output_paths["figure_region"] = generated_region
        output_paths["figure_city_delta"] = generated_city

    summary_payload = build_summary_payload(
        instance_root=instance_root,
        embedding_dir=embedding_dir,
        output_dir=output_dir,
        n_patches=n_patches,
        embedding_dim=embedding_dim,
        group_rows=group_rows,
        decision_rows=decision_rows,
        main_conclusion=main_conclusion,
        output_paths=output_paths,
        args=args,
    )

    log("STEP", "Writing Criterion 2 outputs.")

    write_csv(
        patchwise_csv,
        patchwise_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "patch_index",
            "patch_id",
            "city",
            "region",
            "label_binary",
            "label_name",
            "label_positive_pixels",
            "label_positive_percent",
            "label_density_bin",
            "cosine_s2_snap",
            "cosine_s2_rtc",
            "delta_rtc_minus_snap",
            "higher_similarity_variant",
        ],
    )

    write_csv(
        group_summary_csv,
        group_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "group_type",
            "group_value",
            "n",
            "positive_count",
            "empty_count",

            "snap_mean",
            "snap_std",
            "snap_median",
            "snap_q25",
            "snap_q75",
            "snap_min",
            "snap_max",

            "rtc_mean",
            "rtc_std",
            "rtc_median",
            "rtc_q25",
            "rtc_q75",
            "rtc_min",
            "rtc_max",

            "delta_rtc_minus_snap_mean",
            "delta_rtc_minus_snap_std",
            "delta_rtc_minus_snap_median",
            "delta_rtc_minus_snap_q25",
            "delta_rtc_minus_snap_q75",
            "delta_rtc_minus_snap_min",
            "delta_rtc_minus_snap_max",

            "rtc_wins",
            "snap_wins",
            "ties",
            "rtc_win_percent",
            "snap_win_percent",
            "tie_percent",
            "decision",
        ],
    )

    write_csv(
        decision_csv,
        decision_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "group_type",
            "group_value",
            "n",
            "snap_mean",
            "rtc_mean",
            "delta_rtc_minus_snap_mean",
            "rtc_wins",
            "snap_wins",
            "ties",
            "rtc_win_percent",
            "snap_win_percent",
            "decision",
        ],
    )

    write_json(json_path, summary_payload, overwrite=bool(args.overwrite))

    write_markdown(
        md_path,
        summary=summary_payload,
        group_rows=group_rows,
        decision_rows=decision_rows,
        output_paths=output_paths,
        overwrite=bool(args.overwrite),
    )

    log("OK", f"Wrote patchwise CSV:     {path_to_str(patchwise_csv)}")
    log("OK", f"Wrote group summary CSV: {path_to_str(group_summary_csv)}")
    log("OK", f"Wrote decision CSV:      {path_to_str(decision_csv)}")
    log("OK", f"Wrote JSON:              {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:          {path_to_str(md_path)}")

    if output_paths["figure_histogram"] is not None:
        log("OK", f"Wrote histogram figure:  {path_to_str(output_paths['figure_histogram'])}")

    if output_paths["figure_region"] is not None:
        log("OK", f"Wrote region figure:     {path_to_str(output_paths['figure_region'])}")

    if output_paths["figure_city_delta"] is not None:
        log("OK", f"Wrote city delta figure: {path_to_str(output_paths['figure_city_delta'])}")

    log("STEP", "Final Criterion 2 summary.")
    log("OK", "Status: passed")
    log("OK", f"Patches: {n_patches}")
    log("OK", f"Embedding dimension: {embedding_dim}")
    log("OK", f"Main conclusion: {main_conclusion}")


if __name__ == "__main__":
    main()