#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
14_compute_patch_favela_distribution_statistics_224.py

Compute favela-label distribution statistics over the Instance C patch set.

Purpose
-------
This script computes descriptive statistics about the distribution of favela
labels inside the 224x224 patches used by the CROMA pipeline.

It answers questions such as:

    - How many patches contain favela pixels?
    - What percentage of patches are empty?
    - What is the mean/median/std of favela coverage per patch?
    - How are favela-positive pixels distributed by city and region?
    - Are most positive patches tiny, sparse, or dense?
    - Which cities/regions contain the highest favela patch density?

Important
---------
The CROMA comparison manifest contains one row per patch per modality.
For example, the same patch appears as:

    s2
    s1_snap_vv_vh
    s1_rtc_vv_vh
    s2_s1_snap_vv_vh
    s2_s1_rtc_vv_vh

Therefore, this script deduplicates patches first using patch_id where possible.
This avoids counting each patch five times.

Default input
-------------
<instance-root>/metadata/croma_probing/
    croma_comparison_manifest_ps224_st112_cover.csv

Main outputs
------------
<instance-root>/metadata/croma_probing/patch_favela_distribution/

    patch_favela_distribution_unique_patches_ps224_st112_cover.csv
    patch_favela_distribution_overall_stats_ps224_st112_cover.csv
    patch_favela_distribution_group_stats_ps224_st112_cover.csv
    patch_favela_distribution_density_bins_ps224_st112_cover.csv
    patch_favela_distribution_summary_ps224_st112_cover.json
    patch_favela_distribution_summary_ps224_st112_cover.md

Optional figures
----------------
    figures/patch_favela_percent_histogram_all_ps224_st112_cover.png
    figures/patch_favela_percent_histogram_positive_only_ps224_st112_cover.png
    figures/patch_favela_positive_patch_percent_by_region_ps224_st112_cover.png
    figures/patch_favela_positive_patch_percent_by_city_ps224_st112_cover.png
    figures/patch_favela_mean_percent_by_city_ps224_st112_cover.png

Example
-------
python src/croma_probing/14_compute_patch_favela_distribution_statistics_224.py `
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------
# Logging and utilities
# ---------------------------------------------------------------------

def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    log("ERROR", message)
    raise SystemExit(exit_code)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_to_str(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return str(path).replace("\\", "/")


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
# CSV / JSON / Markdown
# ---------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        fail(f"Input CSV does not exist: {path_to_str(path)}")

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
    overall_rows: List[Dict[str, object]],
    group_rows: List[Dict[str, object]],
    bin_rows: List[Dict[str, object]],
    output_paths: Dict[str, Optional[Path]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    overall = overall_rows[0]

    region_rows = [
        row for row in group_rows
        if row["group_type"] == "region"
    ]

    city_rows = [
        row for row in group_rows
        if row["group_type"] == "city"
    ]

    top_positive_cities = sorted(
        city_rows,
        key=lambda r: safe_float(r["positive_patch_percent"]),
        reverse=True,
    )[:10]

    top_mean_cities = sorted(
        city_rows,
        key=lambda r: safe_float(r["label_positive_percent_mean"]),
        reverse=True,
    )[:10]

    lines: List[str] = []

    lines.append("# Patch-Level Favela Label Distribution Statistics")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Manifest path: `{summary['manifest_path']}`")
    lines.append(f"- Output directory: `{summary['output_dir']}`")
    lines.append(f"- Patch size: `{summary['patch_size']} x {summary['patch_size']}`")
    lines.append(f"- Patch area: `{summary['patch_area_pixels']}` pixels")
    lines.append(f"- Unique patches: `{summary['n_unique_patches']}`")
    lines.append(f"- Manifest rows read: `{summary['n_manifest_rows_read']}`")
    lines.append(f"- Duplicate manifest rows removed: `{summary['n_duplicate_manifest_rows_removed']}`")
    lines.append("")

    lines.append("### Main conclusion")
    lines.append("")
    lines.append(summary["main_conclusion"])
    lines.append("")

    lines.append("## Overall patch distribution")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| total patches | {overall['n_patches']} |")
    lines.append(f"| favela-positive patches | {overall['n_positive_patches']} |")
    lines.append(f"| empty patches | {overall['n_empty_patches']} |")
    lines.append(f"| favela-positive patch percent | {overall['positive_patch_percent']} |")
    lines.append(f"| total favela pixels in patches | {overall['total_label_positive_pixels']} |")
    lines.append(f"| total pixels across patches | {overall['total_patch_pixels']} |")
    lines.append(f"| favela pixel percent across patches | {overall['pixel_positive_percent']} |")
    lines.append(f"| mean favela coverage per patch (%) | {overall['label_positive_percent_mean']} |")
    lines.append(f"| median favela coverage per patch (%) | {overall['label_positive_percent_median']} |")
    lines.append(f"| std favela coverage per patch (%) | {overall['label_positive_percent_std']} |")
    lines.append(f"| p95 favela coverage per patch (%) | {overall['label_positive_percent_p95']} |")
    lines.append(f"| p99 favela coverage per patch (%) | {overall['label_positive_percent_p99']} |")
    lines.append("")

    lines.append("## Density-bin distribution")
    lines.append("")
    lines.append("| density bin | n patches | patch percent | mean coverage (%) | total favela pixels |")
    lines.append("|---|---:|---:|---:|---:|")

    for row in bin_rows:
        lines.append(
            f"| {row['density_bin']} | "
            f"{row['n_patches']} | "
            f"{row['patch_percent']} | "
            f"{row['label_positive_percent_mean']} | "
            f"{row['total_label_positive_pixels']} |"
        )

    lines.append("")
    lines.append("## Region-level distribution")
    lines.append("")
    lines.append("| region | n patches | positive patches | positive patch % | mean coverage % | median coverage % | pixel positive % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for row in sorted(region_rows, key=lambda r: str(r["group_value"])):
        lines.append(
            f"| {row['group_value']} | "
            f"{row['n_patches']} | "
            f"{row['n_positive_patches']} | "
            f"{row['positive_patch_percent']} | "
            f"{row['label_positive_percent_mean']} | "
            f"{row['label_positive_percent_median']} | "
            f"{row['pixel_positive_percent']} |"
        )

    lines.append("")
    lines.append("## Top cities by favela-positive patch percentage")
    lines.append("")
    lines.append("| city | n patches | positive patch % | mean coverage % | pixel positive % |")
    lines.append("|---|---:|---:|---:|---:|")

    for row in top_positive_cities:
        lines.append(
            f"| {row['group_value']} | "
            f"{row['n_patches']} | "
            f"{row['positive_patch_percent']} | "
            f"{row['label_positive_percent_mean']} | "
            f"{row['pixel_positive_percent']} |"
        )

    lines.append("")
    lines.append("## Top cities by mean favela coverage")
    lines.append("")
    lines.append("| city | n patches | positive patch % | mean coverage % | median coverage % |")
    lines.append("|---|---:|---:|---:|---:|")

    for row in top_mean_cities:
        lines.append(
            f"| {row['group_value']} | "
            f"{row['n_patches']} | "
            f"{row['positive_patch_percent']} | "
            f"{row['label_positive_percent_mean']} | "
            f"{row['label_positive_percent_median']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("These statistics describe the label distribution of the exact patch set used by the CROMA comparison. A favela-positive patch is defined as a patch with at least one favela-labelled pixel. The percentage coverage is computed as the number of favela pixels divided by the patch area.")
    lines.append("")
    lines.append("The most important values are the positive-patch percentage and the distribution of label-positive percent. If many positive patches have very low coverage, the task is difficult because the positive class often appears as a small spatial fraction of the patch.")
    lines.append("")
    lines.append("These are patch-level statistics, not unique-pixel full-raster statistics. Because the patch grid uses stride 112 with patch size 224, neighbouring patches overlap. Therefore, this report describes the ML/CROMA patch distribution rather than the unique-area distribution.")
    lines.append("")

    if output_paths.get("histogram_all") is not None:
        lines.append("## Optional generated figures")
        lines.append("")
        for key in [
            "histogram_all",
            "histogram_positive_only",
            "region_positive_patch_percent",
            "city_positive_patch_percent",
            "city_mean_percent",
        ]:
            if output_paths.get(key) is not None:
                lines.append(f"- `{key}`: `{path_to_str(output_paths[key])}`")
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
# Input path and column handling
# ---------------------------------------------------------------------

def default_manifest_path(instance_root: Path, stem: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / f"croma_comparison_manifest_{stem}.csv"
    )


def find_column(
    fieldnames: Sequence[str],
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    lower_to_original = {name.lower(): name for name in fieldnames}

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    if required:
        fail(
            "Could not find required column. Tried candidates:\n"
            f"  {candidates}\n"
            f"Available columns:\n"
            f"  {list(fieldnames)}"
        )

    return None


def make_density_bin(percent: float) -> str:
    p = safe_float(percent)

    if p <= 0:
        return "empty_0"
    if p <= 0.1:
        return "tiny_0_0p1"
    if p <= 1.0:
        return "very_low_0p1_1"
    if p <= 5.0:
        return "low_1_5"
    if p <= 10.0:
        return "moderate_5_10"
    if p <= 25.0:
        return "medium_10_25"
    if p <= 50.0:
        return "high_25_50"
    return "very_high_50_100"


DENSITY_BIN_ORDER = {
    "empty_0": 0,
    "tiny_0_0p1": 1,
    "very_low_0p1_1": 2,
    "low_1_5": 3,
    "moderate_5_10": 4,
    "medium_10_25": 5,
    "high_25_50": 6,
    "very_high_50_100": 7,
}


def load_unique_patch_rows(
    manifest_path: Path,
    *,
    patch_size: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows = read_csv_rows(manifest_path)
    fieldnames = list(rows[0].keys())

    columns = {
        "patch_id": find_column(fieldnames, ["patch_id", "patch_uid", "id"], required=False),
        "city": find_column(fieldnames, ["city"]),
        "region": find_column(fieldnames, ["region"], required=False),
        "split": find_column(fieldnames, ["split", "fold_split", "dataset_split"], required=False),
        "row_off": find_column(
            fieldnames,
            ["row_off", "row_start", "window_row_off", "patch_row_off", "row", "y_off", "y"],
            required=False,
        ),
        "col_off": find_column(
            fieldnames,
            ["col_off", "col_start", "window_col_off", "patch_col_off", "col", "x_off", "x"],
            required=False,
        ),
        "label_binary": find_column(
            fieldnames,
            ["label_binary", "binary_label", "has_favela", "is_positive"],
            required=False,
        ),
        "label_positive_pixels": find_column(
            fieldnames,
            [
                "label_positive_pixels",
                "positive_pixels",
                "favela_pixels",
                "label_sum",
                "mask_positive_pixels",
            ],
            required=False,
        ),
        "label_positive_percent": find_column(
            fieldnames,
            [
                "label_positive_percent",
                "positive_percent",
                "favela_percent",
                "label_positive_pct",
                "mask_positive_percent",
            ],
            required=False,
        ),
        "label_density_bin": find_column(
            fieldnames,
            ["label_density_bin", "density_bin", "positive_density_bin"],
            required=False,
        ),
    }

    if columns["patch_id"] is None and (
        columns["city"] is None or columns["row_off"] is None or columns["col_off"] is None
    ):
        fail(
            "Could not deduplicate patches safely. Need either patch_id, "
            "or city + row/col offsets."
        )

    if columns["label_positive_pixels"] is None and columns["label_positive_percent"] is None:
        fail(
            "Could not find label-positive pixel or percent columns. "
            "Need one of: label_positive_pixels, label_positive_percent."
        )

    patch_area = int(patch_size) * int(patch_size)

    unique: Dict[str, Dict[str, object]] = {}
    duplicate_count = 0
    mismatch_count = 0

    for row in rows:
        city = str(row[columns["city"]]).strip()

        region = (
            str(row[columns["region"]]).strip()
            if columns["region"] is not None
            else "unknown"
        )

        split = (
            str(row[columns["split"]]).strip()
            if columns["split"] is not None
            else "unknown"
        )

        row_off = (
            safe_int(row[columns["row_off"]])
            if columns["row_off"] is not None
            else -1
        )

        col_off = (
            safe_int(row[columns["col_off"]])
            if columns["col_off"] is not None
            else -1
        )

        if columns["patch_id"] is not None:
            patch_id = str(row[columns["patch_id"]]).strip()
            patch_key = patch_id
        else:
            patch_id = f"{city}_r{row_off}_c{col_off}"
            patch_key = patch_id

        if columns["label_positive_pixels"] is not None:
            label_positive_pixels = safe_int(row[columns["label_positive_pixels"]])
            label_positive_percent = 100.0 * label_positive_pixels / patch_area
        else:
            label_positive_percent = safe_float(row[columns["label_positive_percent"]])
            label_positive_pixels = int(round(label_positive_percent / 100.0 * patch_area))

        if columns["label_positive_percent"] is not None:
            label_positive_percent = safe_float(row[columns["label_positive_percent"]])

        label_binary = (
            safe_int(row[columns["label_binary"]])
            if columns["label_binary"] is not None
            else int(label_positive_pixels > 0)
        )

        if label_positive_pixels > 0:
            label_binary = 1

        density_bin = (
            str(row[columns["label_density_bin"]]).strip()
            if columns["label_density_bin"] is not None
            else make_density_bin(label_positive_percent)
        )

        current = {
            "patch_key": patch_key,
            "patch_id": patch_id,
            "city": city,
            "region": region,
            "split": split,
            "row_off": row_off,
            "col_off": col_off,
            "label_binary": int(label_binary),
            "label_positive_pixels": int(label_positive_pixels),
            "label_positive_percent": round_float(label_positive_percent, 10),
            "label_density_bin": density_bin,
            "patch_area_pixels": patch_area,
        }

        if patch_key in unique:
            duplicate_count += 1

            prev = unique[patch_key]

            if (
                int(prev["label_positive_pixels"]) != int(current["label_positive_pixels"])
                or int(prev["label_binary"]) != int(current["label_binary"])
            ):
                mismatch_count += 1

            continue

        unique[patch_key] = current

    unique_rows = list(unique.values())
    unique_rows = sorted(
        unique_rows,
        key=lambda r: (str(r["city"]), safe_int(r["row_off"]), safe_int(r["col_off"]), str(r["patch_id"])),
    )

    metadata = {
        "n_manifest_rows_read": len(rows),
        "n_unique_patches": len(unique_rows),
        "n_duplicate_manifest_rows_removed": duplicate_count,
        "n_duplicate_label_mismatches": mismatch_count,
        "columns_used": columns,
    }

    if mismatch_count > 0:
        log(
            "WARN",
            f"Found {mismatch_count} duplicated patch IDs with label mismatches. "
            "Using first occurrence.",
        )

    return unique_rows, metadata


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def summarize_numeric(values: Sequence[float]) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    return {
        "mean": round_float(float(np.mean(arr)), 8),
        "median": round_float(float(np.median(arr)), 8),
        "std": round_float(std, 8),
        "min": round_float(float(np.min(arr)), 8),
        "p01": round_float(float(np.quantile(arr, 0.01)), 8),
        "p05": round_float(float(np.quantile(arr, 0.05)), 8),
        "p10": round_float(float(np.quantile(arr, 0.10)), 8),
        "p25": round_float(float(np.quantile(arr, 0.25)), 8),
        "p75": round_float(float(np.quantile(arr, 0.75)), 8),
        "p90": round_float(float(np.quantile(arr, 0.90)), 8),
        "p95": round_float(float(np.quantile(arr, 0.95)), 8),
        "p99": round_float(float(np.quantile(arr, 0.99)), 8),
        "max": round_float(float(np.max(arr)), 8),
    }


def summarize_patch_group(
    *,
    group_type: str,
    group_value: str,
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    n = len(rows)

    if n == 0:
        fail(f"Cannot summarize empty group: {group_type}={group_value}")

    patch_area = safe_int(rows[0]["patch_area_pixels"])

    label_pixels = np.asarray(
        [safe_int(row["label_positive_pixels"]) for row in rows],
        dtype=np.float64,
    )

    label_percent = np.asarray(
        [safe_float(row["label_positive_percent"]) for row in rows],
        dtype=np.float64,
    )

    binary = np.asarray(
        [safe_int(row["label_binary"]) for row in rows],
        dtype=np.int64,
    )

    positive_mask = binary == 1

    n_positive = int(np.count_nonzero(positive_mask))
    n_empty = int(n - n_positive)

    total_label_positive_pixels = int(np.sum(label_pixels))
    total_patch_pixels = int(n * patch_area)

    pixel_positive_percent = (
        100.0 * total_label_positive_pixels / total_patch_pixels
        if total_patch_pixels > 0
        else 0.0
    )

    all_stats = summarize_numeric(label_percent.tolist())

    positive_only_percent = label_percent[positive_mask]

    positive_stats = summarize_numeric(positive_only_percent.tolist())

    row = {
        "group_type": group_type,
        "group_value": group_value,
        "n_patches": int(n),
        "n_positive_patches": int(n_positive),
        "n_empty_patches": int(n_empty),
        "positive_patch_percent": round_float(100.0 * n_positive / n if n else 0.0, 8),
        "empty_patch_percent": round_float(100.0 * n_empty / n if n else 0.0, 8),
        "total_label_positive_pixels": total_label_positive_pixels,
        "total_patch_pixels": total_patch_pixels,
        "pixel_positive_percent": round_float(pixel_positive_percent, 8),

        "label_positive_percent_mean": all_stats["mean"],
        "label_positive_percent_median": all_stats["median"],
        "label_positive_percent_std": all_stats["std"],
        "label_positive_percent_min": all_stats["min"],
        "label_positive_percent_p01": all_stats["p01"],
        "label_positive_percent_p05": all_stats["p05"],
        "label_positive_percent_p10": all_stats["p10"],
        "label_positive_percent_p25": all_stats["p25"],
        "label_positive_percent_p75": all_stats["p75"],
        "label_positive_percent_p90": all_stats["p90"],
        "label_positive_percent_p95": all_stats["p95"],
        "label_positive_percent_p99": all_stats["p99"],
        "label_positive_percent_max": all_stats["max"],

        "positive_only_label_percent_mean": positive_stats["mean"],
        "positive_only_label_percent_median": positive_stats["median"],
        "positive_only_label_percent_std": positive_stats["std"],
        "positive_only_label_percent_min": positive_stats["min"],
        "positive_only_label_percent_p25": positive_stats["p25"],
        "positive_only_label_percent_p75": positive_stats["p75"],
        "positive_only_label_percent_p95": positive_stats["p95"],
        "positive_only_label_percent_max": positive_stats["max"],
    }

    return row


def build_overall_stats(unique_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        summarize_patch_group(
            group_type="overall",
            group_value="all_patches",
            rows=unique_rows,
        )
    ]


def build_group_stats(unique_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    group_fields = [
        ("region", "region"),
        ("city", "city"),
        ("split", "split"),
        ("label_density_bin", "label_density_bin"),
    ]

    for group_type, field in group_fields:
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)

        for row in unique_rows:
            value = str(row.get(field, "unknown")).strip()
            if value == "":
                value = "unknown"
            grouped[value].append(row)

        for value in sorted(grouped.keys()):
            rows.append(
                summarize_patch_group(
                    group_type=group_type,
                    group_value=value,
                    rows=grouped[value],
                )
            )

    return rows


def build_density_bin_rows(unique_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    total_patches = len(unique_rows)

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for row in unique_rows:
        bin_name = make_density_bin(safe_float(row["label_positive_percent"]))
        grouped[bin_name].append(row)

    rows: List[Dict[str, object]] = []

    for bin_name in sorted(grouped.keys(), key=lambda b: DENSITY_BIN_ORDER.get(b, 999)):
        bin_rows = grouped[bin_name]
        stat = summarize_patch_group(
            group_type="density_bin",
            group_value=bin_name,
            rows=bin_rows,
        )

        rows.append(
            {
                "density_bin": bin_name,
                "bin_order": DENSITY_BIN_ORDER.get(bin_name, 999),
                "n_patches": stat["n_patches"],
                "patch_percent": round_float(100.0 * safe_int(stat["n_patches"]) / total_patches, 8),
                "n_positive_patches": stat["n_positive_patches"],
                "total_label_positive_pixels": stat["total_label_positive_pixels"],
                "label_positive_percent_mean": stat["label_positive_percent_mean"],
                "label_positive_percent_median": stat["label_positive_percent_median"],
                "label_positive_percent_min": stat["label_positive_percent_min"],
                "label_positive_percent_max": stat["label_positive_percent_max"],
            }
        )

    return rows


def build_main_conclusion(overall_rows: List[Dict[str, object]]) -> str:
    row = overall_rows[0]

    n = safe_int(row["n_patches"])
    n_pos = safe_int(row["n_positive_patches"])
    pos_patch_percent = safe_float(row["positive_patch_percent"])
    pixel_positive_percent = safe_float(row["pixel_positive_percent"])
    median = safe_float(row["label_positive_percent_median"])
    pos_median = safe_float(row["positive_only_label_percent_median"])

    return (
        f"The patch set contains {n} unique patches, of which {n_pos} "
        f"({round_float(pos_patch_percent, 4)}%) contain at least one favela-labelled pixel. "
        f"Across all patches, favela pixels represent {round_float(pixel_positive_percent, 4)}% "
        f"of patch pixels. The median favela coverage across all patches is "
        f"{round_float(median, 4)}%, while the median among positive patches only is "
        f"{round_float(pos_median, 4)}%. This indicates how sparse or dense the favela signal is "
        f"inside the CROMA patch dataset."
    )


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def make_histogram_all(
    *,
    unique_rows: List[Dict[str, object]],
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        log("WARN", "matplotlib is not installed; skipping figures.")
        return None

    ensure_output_can_be_written(output_path, overwrite)

    values = np.asarray(
        [safe_float(row["label_positive_percent"]) for row in unique_rows],
        dtype=np.float64,
    )

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.hist(values, bins=80)
    ax.set_title("Favela coverage percentage across all patches")
    ax.set_xlabel("Favela-labelled pixels per patch (%)")
    ax.set_ylabel("Number of patches")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def make_histogram_positive_only(
    *,
    unique_rows: List[Dict[str, object]],
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        return None

    ensure_output_can_be_written(output_path, overwrite)

    values = np.asarray(
        [
            safe_float(row["label_positive_percent"])
            for row in unique_rows
            if safe_float(row["label_positive_percent"]) > 0
        ],
        dtype=np.float64,
    )

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.hist(values, bins=80)
    ax.set_title("Favela coverage percentage across positive patches only")
    ax.set_xlabel("Favela-labelled pixels per positive patch (%)")
    ax.set_ylabel("Number of positive patches")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def make_region_bar(
    *,
    group_rows: List[Dict[str, object]],
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        return None

    rows = [
        row for row in group_rows
        if row["group_type"] == "region"
    ]

    if not rows:
        return None

    rows = sorted(rows, key=lambda r: str(r["group_value"]))

    ensure_output_can_be_written(output_path, overwrite)

    labels = [str(row["group_value"]) for row in rows]
    values = [safe_float(row["positive_patch_percent"]) for row in rows]

    x = list(range(len(labels)))

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    ax.bar(x, values)
    ax.set_title("Favela-positive patch percentage by region")
    ax.set_ylabel("Positive patches (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def make_city_bar(
    *,
    group_rows: List[Dict[str, object]],
    output_path: Path,
    overwrite: bool,
    metric: str,
    title: str,
    ylabel: str,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        return None

    rows = [
        row for row in group_rows
        if row["group_type"] == "city"
    ]

    if not rows:
        return None

    rows = sorted(rows, key=lambda r: safe_float(r[metric]), reverse=True)

    ensure_output_can_be_written(output_path, overwrite)

    labels = [str(row["group_value"]) for row in rows]
    values = [safe_float(row[metric]) for row in rows]

    x = list(range(len(labels)))

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    ax.bar(x, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute favela-label distribution statistics over 224x224 patches."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/croma_comparison_manifest_<stem>.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/patch_favela_distribution.",
    )

    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--edge-mode", choices=["cover", "drop"], default="cover")

    parser.add_argument(
        "--make-figures",
        action="store_true",
        help="Generate figures if matplotlib is installed.",
    )

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root
    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    manifest_path: Path = args.manifest_path or default_manifest_path(instance_root, stem)

    output_dir: Path = args.output_dir or (
        instance_root
        / "metadata"
        / "croma_probing"
        / "patch_favela_distribution"
    )

    unique_csv = output_dir / f"patch_favela_distribution_unique_patches_{stem}.csv"
    overall_csv = output_dir / f"patch_favela_distribution_overall_stats_{stem}.csv"
    group_csv = output_dir / f"patch_favela_distribution_group_stats_{stem}.csv"
    bins_csv = output_dir / f"patch_favela_distribution_density_bins_{stem}.csv"
    json_path = output_dir / f"patch_favela_distribution_summary_{stem}.json"
    md_path = output_dir / f"patch_favela_distribution_summary_{stem}.md"

    figure_hist_all: Optional[Path] = None
    figure_hist_positive: Optional[Path] = None
    figure_region: Optional[Path] = None
    figure_city_positive: Optional[Path] = None
    figure_city_mean: Optional[Path] = None

    if args.make_figures:
        figure_dir = output_dir / "figures"
        figure_hist_all = figure_dir / f"patch_favela_percent_histogram_all_{stem}.png"
        figure_hist_positive = figure_dir / f"patch_favela_percent_histogram_positive_only_{stem}.png"
        figure_region = figure_dir / f"patch_favela_positive_patch_percent_by_region_{stem}.png"
        figure_city_positive = figure_dir / f"patch_favela_positive_patch_percent_by_city_{stem}.png"
        figure_city_mean = figure_dir / f"patch_favela_mean_percent_by_city_{stem}.png"

    output_paths: Dict[str, Optional[Path]] = {
        "unique_patches_csv": unique_csv,
        "overall_stats_csv": overall_csv,
        "group_stats_csv": group_csv,
        "density_bins_csv": bins_csv,
        "json": json_path,
        "markdown": md_path,
        "histogram_all": figure_hist_all,
        "histogram_positive_only": figure_hist_positive,
        "region_positive_patch_percent": figure_region,
        "city_positive_patch_percent": figure_city_positive,
        "city_mean_percent": figure_city_mean,
    }

    log("STEP", "Computing patch-level favela label distribution statistics.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Manifest:      {path_to_str(manifest_path)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")
    log("INFO", f"Stem:          {stem}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not manifest_path.exists():
        fail(f"CROMA comparison manifest does not exist: {path_to_str(manifest_path)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    unique_rows, metadata = load_unique_patch_rows(
        manifest_path,
        patch_size=int(args.patch_size),
    )

    log("OK", f"Manifest rows read: {metadata['n_manifest_rows_read']}")
    log("OK", f"Unique patches: {metadata['n_unique_patches']}")
    log("OK", f"Duplicate manifest rows removed: {metadata['n_duplicate_manifest_rows_removed']}")

    overall_rows = build_overall_stats(unique_rows)
    group_rows = build_group_stats(unique_rows)
    bin_rows = build_density_bin_rows(unique_rows)

    main_conclusion = build_main_conclusion(overall_rows)

    if args.make_figures:
        output_paths["histogram_all"] = make_histogram_all(
            unique_rows=unique_rows,
            output_path=figure_hist_all,
            overwrite=bool(args.overwrite),
        )

        output_paths["histogram_positive_only"] = make_histogram_positive_only(
            unique_rows=unique_rows,
            output_path=figure_hist_positive,
            overwrite=bool(args.overwrite),
        )

        output_paths["region_positive_patch_percent"] = make_region_bar(
            group_rows=group_rows,
            output_path=figure_region,
            overwrite=bool(args.overwrite),
        )

        output_paths["city_positive_patch_percent"] = make_city_bar(
            group_rows=group_rows,
            output_path=figure_city_positive,
            overwrite=bool(args.overwrite),
            metric="positive_patch_percent",
            title="Favela-positive patch percentage by city",
            ylabel="Positive patches (%)",
        )

        output_paths["city_mean_percent"] = make_city_bar(
            group_rows=group_rows,
            output_path=figure_city_mean,
            overwrite=bool(args.overwrite),
            metric="label_positive_percent_mean",
            title="Mean favela coverage percentage by city",
            ylabel="Mean favela coverage per patch (%)",
        )

    summary_payload = {
        "created_utc": now_utc(),
        "status": "passed",
        "instance_root": path_to_str(instance_root),
        "manifest_path": path_to_str(manifest_path),
        "output_dir": path_to_str(output_dir),
        "patch_size": int(args.patch_size),
        "stride": int(args.stride),
        "edge_mode": str(args.edge_mode),
        "patch_area_pixels": int(args.patch_size) * int(args.patch_size),
        "n_manifest_rows_read": int(metadata["n_manifest_rows_read"]),
        "n_unique_patches": int(metadata["n_unique_patches"]),
        "n_duplicate_manifest_rows_removed": int(metadata["n_duplicate_manifest_rows_removed"]),
        "n_duplicate_label_mismatches": int(metadata["n_duplicate_label_mismatches"]),
        "main_conclusion": main_conclusion,
        "columns_used": {
            key: "" if value is None else str(value)
            for key, value in metadata["columns_used"].items()
        },
        "outputs": {
            key: "" if value is None else path_to_str(value)
            for key, value in output_paths.items()
        },
    }

    log("STEP", "Writing patch favela distribution outputs.")

    write_csv(
        unique_csv,
        unique_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "patch_key",
            "patch_id",
            "city",
            "region",
            "split",
            "row_off",
            "col_off",
            "label_binary",
            "label_positive_pixels",
            "label_positive_percent",
            "label_density_bin",
            "patch_area_pixels",
        ],
    )

    write_csv(
        overall_csv,
        overall_rows,
        overwrite=bool(args.overwrite),
    )

    write_csv(
        group_csv,
        group_rows,
        overwrite=bool(args.overwrite),
    )

    write_csv(
        bins_csv,
        bin_rows,
        overwrite=bool(args.overwrite),
    )

    write_json(json_path, summary_payload, overwrite=bool(args.overwrite))

    write_markdown(
        md_path,
        summary=summary_payload,
        overall_rows=overall_rows,
        group_rows=group_rows,
        bin_rows=bin_rows,
        output_paths=output_paths,
        overwrite=bool(args.overwrite),
    )

    log("OK", f"Wrote unique patches CSV: {path_to_str(unique_csv)}")
    log("OK", f"Wrote overall stats CSV:  {path_to_str(overall_csv)}")
    log("OK", f"Wrote group stats CSV:    {path_to_str(group_csv)}")
    log("OK", f"Wrote density bins CSV:   {path_to_str(bins_csv)}")
    log("OK", f"Wrote JSON:               {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:           {path_to_str(md_path)}")

    for key in [
        "histogram_all",
        "histogram_positive_only",
        "region_positive_patch_percent",
        "city_positive_patch_percent",
        "city_mean_percent",
    ]:
        if output_paths.get(key) is not None:
            log("OK", f"Wrote figure {key}: {path_to_str(output_paths[key])}")

    log("STEP", "Final patch favela distribution summary.")
    log("OK", "Status: passed")
    log("OK", f"Unique patches: {len(unique_rows)}")
    log("OK", f"Main conclusion: {main_conclusion}")


if __name__ == "__main__":
    main()