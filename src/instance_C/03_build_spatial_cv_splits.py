#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_build_spatial_cv_splits.py

Build spatial cross-validation splits for Instance C.

Input:
    metadata/instance_C_patches/patch_metadata_ps224_st112_cover.csv

Outputs:
    metadata/instance_C_splits/
        leave_one_city_out/
            fold_test_<city>/
                train.csv
                val.csv
                test.csv
                summary.json
                summary.md
            leave_one_city_out_summary.csv
            leave_one_city_out_summary.json
            leave_one_city_out_summary.md

        leave_one_region_out/
            fold_test_<region>/
                train.csv
                val.csv
                test.csv
                summary.json
                summary.md
            leave_one_region_out_summary.csv
            leave_one_region_out_summary.json
            leave_one_region_out_summary.md

This script is designed for the updated experimental plan:

    Q1. RTC vs SNAP-GRD:
        CROMA embeddings/probing.

    Q2. Split strategy:
        CROMA + UPerNet dense segmentation.

No random patch split is created because patches overlap spatially.
Splits are group-based by city or region to avoid leakage.

Example PowerShell command:

python src/instance_C/03_build_spatial_cv_splits.py `
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
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------
# Logging helpers
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


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_name(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


# ---------------------------------------------------------------------
# CSV / JSON / Markdown I/O
# ---------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        fail(f"Input CSV does not exist: {path_to_str(path)}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        fail(f"Input CSV is empty: {path_to_str(path)}")

    return rows


def write_csv_rows(
    path: Path,
    rows: List[Dict[str, object]],
    fieldnames: Sequence[str],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(path: Path, lines: List[str], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_required_columns(rows: List[Dict[str, str]]) -> None:
    required = [
        "patch_id",
        "city",
        "region",
        "patch_label_binary",
        "label_positive_pixels",
        "label_positive_percent",
        "label_density_bin",
        "s2_valid_percent",
        "s1_snap_grd_valid_percent",
        "source_s2_path",
        "source_s1_snap_grd_path",
        "source_label_path",
    ]

    columns = set(rows[0].keys())
    missing = [col for col in required if col not in columns]

    if missing:
        fail(
            "Patch metadata CSV is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def validate_unique_patch_ids(rows: List[Dict[str, str]]) -> None:
    counter = Counter(row["patch_id"] for row in rows)
    duplicates = [patch_id for patch_id, count in counter.items() if count > 1]

    if duplicates:
        shown = "\n".join(f"  - {patch_id}" for patch_id in duplicates[:20])
        extra = "" if len(duplicates) <= 20 else f"\n  ... and {len(duplicates) - 20} more"
        fail(
            "Duplicate patch_id values found. Splits require unique patch IDs:\n"
            f"{shown}{extra}"
        )


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def summarize_rows(rows: List[Dict[str, str]]) -> Dict[str, object]:
    n = len(rows)

    positive_patches = sum(safe_int(row["patch_label_binary"]) for row in rows)
    empty_patches = n - positive_patches

    positive_percent = 100.0 * positive_patches / n if n else 0.0
    empty_percent = 100.0 * empty_patches / n if n else 0.0

    total_label_positive_pixels = sum(
        safe_int(row.get("label_positive_pixels", 0))
        for row in rows
    )

    label_positive_percent_values = [
        safe_float(row.get("label_positive_percent", 0.0))
        for row in rows
    ]

    mean_label_positive_percent = (
        sum(label_positive_percent_values) / n
        if n else 0.0
    )

    max_label_positive_percent = (
        max(label_positive_percent_values)
        if label_positive_percent_values else 0.0
    )

    s2_valid_values = [
        safe_float(row.get("s2_valid_percent", 0.0))
        for row in rows
    ]

    s1_valid_values = [
        safe_float(row.get("s1_snap_grd_valid_percent", 0.0))
        for row in rows
    ]

    min_s2_valid_percent = min(s2_valid_values) if s2_valid_values else 0.0
    min_s1_valid_percent = min(s1_valid_values) if s1_valid_values else 0.0

    density_counts = Counter(row.get("label_density_bin", "UNKNOWN") for row in rows)
    city_counts = Counter(row["city"] for row in rows)
    region_counts = Counter(row["region"] for row in rows)

    cities = sorted(city_counts.keys())
    regions = sorted(region_counts.keys())

    return {
        "patches": n,
        "positive_patches": positive_patches,
        "empty_patches": empty_patches,
        "positive_patch_percent": positive_percent,
        "empty_patch_percent": empty_percent,
        "total_label_positive_pixels": total_label_positive_pixels,
        "mean_label_positive_percent": mean_label_positive_percent,
        "max_label_positive_percent": max_label_positive_percent,
        "min_s2_valid_percent": min_s2_valid_percent,
        "min_s1_snap_grd_valid_percent": min_s1_valid_percent,
        "label_density_bin_counts": {
            "empty": density_counts.get("empty", 0),
            "low": density_counts.get("low", 0),
            "medium": density_counts.get("medium", 0),
            "high": density_counts.get("high", 0),
            "UNKNOWN": density_counts.get("UNKNOWN", 0),
        },
        "cities": cities,
        "regions": regions,
        "city_counts": dict(sorted(city_counts.items())),
        "region_counts": dict(sorted(region_counts.items())),
    }


def summarize_by_city(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[row["city"]].append(row)

    out: Dict[str, Dict[str, object]] = {}

    for city, city_rows in grouped.items():
        summary = summarize_rows(city_rows)
        summary["city"] = city
        summary["region"] = city_rows[0]["region"]
        out[city] = summary

    return dict(sorted(out.items()))


def summarize_by_region(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[row["region"]].append(row)

    out: Dict[str, Dict[str, object]] = {}

    for region, region_rows in grouped.items():
        summary = summarize_rows(region_rows)
        summary["region"] = region
        out[region] = summary

    return dict(sorted(out.items()))


# ---------------------------------------------------------------------
# Validation-selection heuristics
# ---------------------------------------------------------------------

def select_validation_city(
    candidate_cities: Sequence[str],
    city_stats: Dict[str, Dict[str, object]],
    *,
    excluded_cities: Sequence[str],
    target_positive_percent: float,
    target_patch_count: float,
    min_val_patches: int,
    min_val_positive_patches: int,
) -> Tuple[str, Dict[str, object]]:
    """
    Deterministically select one validation city.

    Preference:
        1. city is not excluded
        2. enough patches
        3. enough positive patches
        4. positive patch percent close to global target
        5. patch count close to median city size
        6. stable city-name tie-breaker

    If no candidate satisfies the minimum thresholds, the script falls back to
    the best available non-excluded candidate and records a warning.
    """

    excluded = set(excluded_cities)
    candidates = [city for city in candidate_cities if city not in excluded]

    if not candidates:
        fail("No candidate cities available for validation selection.")

    eligible = []

    for city in candidates:
        stats = city_stats[city]
        patches = int(stats["patches"])
        positives = int(stats["positive_patches"])

        if patches >= min_val_patches and positives >= min_val_positive_patches:
            eligible.append(city)

    used_fallback = False

    if not eligible:
        eligible = candidates
        used_fallback = True

    scored = []

    for city in eligible:
        stats = city_stats[city]

        pos_pct = float(stats["positive_patch_percent"])
        patches = int(stats["patches"])

        score_positive = abs(pos_pct - target_positive_percent)
        score_size = abs(patches - target_patch_count) / max(target_patch_count, 1.0)

        # Positive balance is much more important than city size.
        score = (score_positive, score_size, city)

        scored.append((score, city))

    scored.sort(key=lambda item: item[0])
    selected_city = scored[0][1]

    info = {
        "selected_city": selected_city,
        "used_fallback": used_fallback,
        "candidate_cities": sorted(candidates),
        "eligible_cities": sorted(eligible),
        "selection_score": {
            "positive_percent_distance": float(scored[0][0][0]),
            "patch_count_relative_distance": float(scored[0][0][1]),
        },
        "target_positive_percent": float(target_positive_percent),
        "target_patch_count": float(target_patch_count),
        "min_val_patches": int(min_val_patches),
        "min_val_positive_patches": int(min_val_positive_patches),
    }

    return selected_city, info


def select_validation_cities_per_region(
    candidate_cities: Sequence[str],
    city_stats: Dict[str, Dict[str, object]],
    *,
    excluded_regions: Sequence[str],
    cities_per_region: int,
    target_positive_percent: float,
    target_patch_count: float,
    min_val_patches: int,
    min_val_positive_patches: int,
) -> Tuple[List[str], Dict[str, object]]:
    """
    Deterministically select validation cities for leave-one-region-out.

    Default:
        select one city from each non-held-out region.

    This gives a validation set that is not dominated by a single city/region.
    """

    excluded_regions_set = set(excluded_regions)

    region_to_cities: Dict[str, List[str]] = defaultdict(list)

    for city in candidate_cities:
        region = str(city_stats[city]["region"])

        if region in excluded_regions_set:
            continue

        region_to_cities[region].append(city)

    selected: List[str] = []
    region_details: Dict[str, object] = {}

    for region in sorted(region_to_cities):
        available = sorted(region_to_cities[region])
        region_selected: List[str] = []
        region_candidate_pool = available.copy()

        for _ in range(cities_per_region):
            if not region_candidate_pool:
                break

            chosen, info = select_validation_city(
                candidate_cities=region_candidate_pool,
                city_stats=city_stats,
                excluded_cities=[],
                target_positive_percent=target_positive_percent,
                target_patch_count=target_patch_count,
                min_val_patches=min_val_patches,
                min_val_positive_patches=min_val_positive_patches,
            )

            region_selected.append(chosen)
            selected.append(chosen)
            region_candidate_pool = [
                city for city in region_candidate_pool
                if city != chosen
            ]

        region_details[region] = {
            "available_cities": available,
            "selected_cities": region_selected,
        }

    if not selected:
        fail("No validation cities selected for leave-one-region-out.")

    info = {
        "selected_cities": selected,
        "excluded_regions": sorted(excluded_regions_set),
        "cities_per_region": int(cities_per_region),
        "region_details": region_details,
        "target_positive_percent": float(target_positive_percent),
        "target_patch_count": float(target_patch_count),
        "min_val_patches": int(min_val_patches),
        "min_val_positive_patches": int(min_val_positive_patches),
    }

    return selected, info


# ---------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------

def add_split_columns(
    rows: List[Dict[str, str]],
    *,
    split_family: str,
    fold_id: str,
    split_role: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []

    for row in rows:
        new_row: Dict[str, object] = {
            "split_family": split_family,
            "fold_id": fold_id,
            "split_role": split_role,
        }
        new_row.update(row)
        out.append(new_row)

    return out


def check_no_patch_overlap(
    train_rows: List[Dict[str, str]],
    val_rows: List[Dict[str, str]],
    test_rows: List[Dict[str, str]],
) -> Dict[str, object]:
    train_ids = set(row["patch_id"] for row in train_rows)
    val_ids = set(row["patch_id"] for row in val_rows)
    test_ids = set(row["patch_id"] for row in test_rows)

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    return {
        "train_val_overlap_count": len(train_val_overlap),
        "train_test_overlap_count": len(train_test_overlap),
        "val_test_overlap_count": len(val_test_overlap),
        "has_overlap": bool(train_val_overlap or train_test_overlap or val_test_overlap),
    }


def build_fold_summary(
    *,
    split_family: str,
    fold_id: str,
    test_group_name: str,
    validation_selection: Dict[str, object],
    train_rows: List[Dict[str, str]],
    val_rows: List[Dict[str, str]],
    test_rows: List[Dict[str, str]],
) -> Dict[str, object]:
    train_summary = summarize_rows(train_rows)
    val_summary = summarize_rows(val_rows)
    test_summary = summarize_rows(test_rows)

    overlap = check_no_patch_overlap(train_rows, val_rows, test_rows)

    warnings: List[str] = []

    if overlap["has_overlap"]:
        warnings.append("Patch overlap detected between train/val/test splits.")

    if val_summary["positive_patches"] == 0:
        warnings.append("Validation split has zero positive patches.")

    if test_summary["positive_patches"] == 0:
        warnings.append("Test split has zero positive patches.")

    if train_summary["positive_patches"] == 0:
        warnings.append("Train split has zero positive patches.")

    if val_summary["patches"] == 0:
        warnings.append("Validation split is empty.")

    if test_summary["patches"] == 0:
        warnings.append("Test split is empty.")

    if train_summary["patches"] == 0:
        warnings.append("Train split is empty.")

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_family": split_family,
        "fold_id": fold_id,
        "test_group_name": test_group_name,
        "validation_selection": validation_selection,
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
        "overlap_check": overlap,
        "warnings": warnings,
    }


def build_leave_one_city_out_folds(
    rows: List[Dict[str, str]],
    city_stats: Dict[str, Dict[str, object]],
    *,
    target_positive_percent: float,
    target_patch_count: float,
    min_val_patches: int,
    min_val_positive_patches: int,
) -> List[Dict[str, object]]:
    cities = sorted(city_stats.keys())
    folds: List[Dict[str, object]] = []

    for test_city in cities:
        val_city, val_info = select_validation_city(
            candidate_cities=cities,
            city_stats=city_stats,
            excluded_cities=[test_city],
            target_positive_percent=target_positive_percent,
            target_patch_count=target_patch_count,
            min_val_patches=min_val_patches,
            min_val_positive_patches=min_val_positive_patches,
        )

        train_cities = [
            city for city in cities
            if city not in {test_city, val_city}
        ]

        train_rows = [row for row in rows if row["city"] in set(train_cities)]
        val_rows = [row for row in rows if row["city"] == val_city]
        test_rows = [row for row in rows if row["city"] == test_city]

        fold_id = f"fold_test_{sanitize_name(test_city)}"

        summary = build_fold_summary(
            split_family="leave_one_city_out",
            fold_id=fold_id,
            test_group_name=test_city,
            validation_selection=val_info,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
        )

        folds.append(
            {
                "split_family": "leave_one_city_out",
                "fold_id": fold_id,
                "test_city": test_city,
                "val_city": val_city,
                "train_cities": train_cities,
                "train_rows": train_rows,
                "val_rows": val_rows,
                "test_rows": test_rows,
                "summary": summary,
            }
        )

    return folds


def build_leave_one_region_out_folds(
    rows: List[Dict[str, str]],
    city_stats: Dict[str, Dict[str, object]],
    region_stats: Dict[str, Dict[str, object]],
    *,
    target_positive_percent: float,
    target_patch_count: float,
    min_val_patches: int,
    min_val_positive_patches: int,
    loro_val_cities_per_region: int,
) -> List[Dict[str, object]]:
    regions = sorted(region_stats.keys())
    cities = sorted(city_stats.keys())

    folds: List[Dict[str, object]] = []

    for test_region in regions:
        test_cities = sorted(
            city for city, stats in city_stats.items()
            if stats["region"] == test_region
        )

        candidate_val_cities = [
            city for city in cities
            if city not in set(test_cities)
        ]

        val_cities, val_info = select_validation_cities_per_region(
            candidate_cities=candidate_val_cities,
            city_stats=city_stats,
            excluded_regions=[test_region],
            cities_per_region=loro_val_cities_per_region,
            target_positive_percent=target_positive_percent,
            target_patch_count=target_patch_count,
            min_val_patches=min_val_patches,
            min_val_positive_patches=min_val_positive_patches,
        )

        val_city_set = set(val_cities)
        test_city_set = set(test_cities)

        train_cities = [
            city for city in cities
            if city not in test_city_set and city not in val_city_set
        ]

        train_rows = [row for row in rows if row["city"] in set(train_cities)]
        val_rows = [row for row in rows if row["city"] in val_city_set]
        test_rows = [row for row in rows if row["region"] == test_region]

        fold_id = f"fold_test_{sanitize_name(test_region)}"

        summary = build_fold_summary(
            split_family="leave_one_region_out",
            fold_id=fold_id,
            test_group_name=test_region,
            validation_selection=val_info,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
        )

        folds.append(
            {
                "split_family": "leave_one_region_out",
                "fold_id": fold_id,
                "test_region": test_region,
                "test_cities": test_cities,
                "val_cities": val_cities,
                "train_cities": train_cities,
                "train_rows": train_rows,
                "val_rows": val_rows,
                "test_rows": test_rows,
                "summary": summary,
            }
        )

    return folds


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def fold_summary_to_flat_row(fold: Dict[str, object]) -> Dict[str, object]:
    summary = fold["summary"]

    train = summary["train"]
    val = summary["val"]
    test = summary["test"]
    overlap = summary["overlap_check"]

    row = {
        "split_family": summary["split_family"],
        "fold_id": summary["fold_id"],
        "test_group_name": summary["test_group_name"],

        "train_patches": train["patches"],
        "train_positive_patches": train["positive_patches"],
        "train_positive_patch_percent": round(float(train["positive_patch_percent"]), 8),
        "train_regions": ";".join(train["regions"]),
        "train_cities": ";".join(train["cities"]),

        "val_patches": val["patches"],
        "val_positive_patches": val["positive_patches"],
        "val_positive_patch_percent": round(float(val["positive_patch_percent"]), 8),
        "val_regions": ";".join(val["regions"]),
        "val_cities": ";".join(val["cities"]),

        "test_patches": test["patches"],
        "test_positive_patches": test["positive_patches"],
        "test_positive_patch_percent": round(float(test["positive_patch_percent"]), 8),
        "test_regions": ";".join(test["regions"]),
        "test_cities": ";".join(test["cities"]),

        "train_val_overlap_count": overlap["train_val_overlap_count"],
        "train_test_overlap_count": overlap["train_test_overlap_count"],
        "val_test_overlap_count": overlap["val_test_overlap_count"],

        "warning_count": len(summary["warnings"]),
        "warnings": " | ".join(summary["warnings"]),
    }

    return row


def build_fold_markdown(summary: Dict[str, object]) -> List[str]:
    train = summary["train"]
    val = summary["val"]
    test = summary["test"]

    lines: List[str] = []

    lines.append(f"# Split fold summary: {summary['fold_id']}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Split family: `{summary['split_family']}`")
    lines.append(f"- Test group: `{summary['test_group_name']}`")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append("")

    lines.append("## Split sizes")
    lines.append("")
    lines.append("| split | patches | positive patches | positive patch % | cities | regions |")
    lines.append("|---|---:|---:|---:|---|---|")
    lines.append(
        f"| train | {train['patches']} | {train['positive_patches']} | "
        f"{float(train['positive_patch_percent']):.6f} | "
        f"{', '.join(train['cities'])} | {', '.join(train['regions'])} |"
    )
    lines.append(
        f"| val | {val['patches']} | {val['positive_patches']} | "
        f"{float(val['positive_patch_percent']):.6f} | "
        f"{', '.join(val['cities'])} | {', '.join(val['regions'])} |"
    )
    lines.append(
        f"| test | {test['patches']} | {test['positive_patches']} | "
        f"{float(test['positive_patch_percent']):.6f} | "
        f"{', '.join(test['cities'])} | {', '.join(test['regions'])} |"
    )
    lines.append("")

    lines.append("## Label density bins")
    lines.append("")
    lines.append("| split | empty | low | medium | high |")
    lines.append("|---|---:|---:|---:|---:|")

    for split_name, split_summary in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        density = split_summary["label_density_bin_counts"]
        lines.append(
            f"| {split_name} | "
            f"{density.get('empty', 0)} | "
            f"{density.get('low', 0)} | "
            f"{density.get('medium', 0)} | "
            f"{density.get('high', 0)} |"
        )

    lines.append("")
    lines.append("## Overlap check")
    lines.append("")
    overlap = summary["overlap_check"]
    lines.append(f"- train/val overlap: `{overlap['train_val_overlap_count']}`")
    lines.append(f"- train/test overlap: `{overlap['train_test_overlap_count']}`")
    lines.append(f"- val/test overlap: `{overlap['val_test_overlap_count']}`")
    lines.append(f"- has overlap: `{overlap['has_overlap']}`")
    lines.append("")

    lines.append("## Validation selection")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["validation_selection"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Warnings")
    lines.append("")

    if summary["warnings"]:
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    return lines


def build_family_markdown(
    split_family: str,
    flat_rows: List[Dict[str, object]],
    output_root: Path,
) -> List[str]:
    lines: List[str] = []

    lines.append(f"# {split_family} summary")
    lines.append("")
    lines.append(f"- Output root: `{path_to_str(output_root)}`")
    lines.append(f"- Folds: `{len(flat_rows)}`")
    lines.append("")

    lines.append("## Fold overview")
    lines.append("")
    lines.append(
        "| fold | test group | train patches | val patches | test patches | "
        "train positive % | val positive % | test positive % | warnings |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for row in flat_rows:
        lines.append(
            f"| {row['fold_id']} | "
            f"{row['test_group_name']} | "
            f"{row['train_patches']} | "
            f"{row['val_patches']} | "
            f"{row['test_patches']} | "
            f"{float(row['train_positive_patch_percent']):.6f} | "
            f"{float(row['val_positive_patch_percent']):.6f} | "
            f"{float(row['test_positive_patch_percent']):.6f} | "
            f"{row['warning_count']} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- These are spatial group splits, not random patch splits.")
    lines.append("- Train/validation/test patch IDs are checked for overlap.")
    lines.append("- Positive patch percentage can vary strongly by city and region.")
    lines.append("- For CROMA+UPerNet, validation should be used for model selection and threshold selection only.")
    lines.append("- Test metrics should be reported only after validation-based selection.")

    return lines


def write_fold_outputs(
    fold: Dict[str, object],
    family_dir: Path,
    original_fieldnames: Sequence[str],
    overwrite: bool,
) -> None:
    fold_id = str(fold["fold_id"])
    fold_dir = family_dir / fold_id

    split_family = str(fold["split_family"])

    train_rows = add_split_columns(
        fold["train_rows"],
        split_family=split_family,
        fold_id=fold_id,
        split_role="train",
    )

    val_rows = add_split_columns(
        fold["val_rows"],
        split_family=split_family,
        fold_id=fold_id,
        split_role="val",
    )

    test_rows = add_split_columns(
        fold["test_rows"],
        split_family=split_family,
        fold_id=fold_id,
        split_role="test",
    )

    fieldnames = [
        "split_family",
        "fold_id",
        "split_role",
    ] + list(original_fieldnames)

    write_csv_rows(fold_dir / "train.csv", train_rows, fieldnames, overwrite)
    write_csv_rows(fold_dir / "val.csv", val_rows, fieldnames, overwrite)
    write_csv_rows(fold_dir / "test.csv", test_rows, fieldnames, overwrite)

    summary = fold["summary"]

    write_json(fold_dir / "summary.json", summary, overwrite)
    write_markdown(fold_dir / "summary.md", build_fold_markdown(summary), overwrite)


def write_family_outputs(
    folds: List[Dict[str, object]],
    family_dir: Path,
    split_family: str,
    original_fieldnames: Sequence[str],
    overwrite: bool,
) -> None:
    family_dir.mkdir(parents=True, exist_ok=True)

    flat_rows = [fold_summary_to_flat_row(fold) for fold in folds]

    flat_fieldnames = list(flat_rows[0].keys()) if flat_rows else []

    summary_csv = family_dir / f"{split_family}_summary.csv"
    summary_json = family_dir / f"{split_family}_summary.json"
    summary_md = family_dir / f"{split_family}_summary.md"

    write_csv_rows(summary_csv, flat_rows, flat_fieldnames, overwrite)

    family_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_family": split_family,
        "n_folds": len(folds),
        "folds": [fold["summary"] for fold in folds],
        "outputs": {
            "summary_csv": path_to_str(summary_csv),
            "summary_json": path_to_str(summary_json),
            "summary_md": path_to_str(summary_md),
        },
    }

    write_json(summary_json, family_payload, overwrite)
    write_markdown(summary_md, build_family_markdown(split_family, flat_rows, family_dir), overwrite)

    for fold in folds:
        write_fold_outputs(
            fold=fold,
            family_dir=family_dir,
            original_fieldnames=original_fieldnames,
            overwrite=overwrite,
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build spatial CV splits for Instance C patch metadata."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
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
        "--input-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit patch metadata CSV. "
            "Default: <instance-root>/metadata/instance_C_patches/"
            "patch_metadata_ps<patch-size>_st<stride>_<edge-mode>.csv"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output root. "
            "Default: <instance-root>/metadata/instance_C_splits"
        ),
    )

    parser.add_argument(
        "--split-family",
        choices=["both", "leave_one_city_out", "leave_one_region_out"],
        default="both",
        help="Which split family to build. Default: both.",
    )

    parser.add_argument(
        "--min-val-patches",
        type=int,
        default=100,
        help="Minimum preferred validation city patch count. Default: 100.",
    )

    parser.add_argument(
        "--min-val-positive-patches",
        type=int,
        default=20,
        help="Minimum preferred validation city positive-patch count. Default: 20.",
    )

    parser.add_argument(
        "--loro-val-cities-per-region",
        type=int,
        default=1,
        help=(
            "For leave-one-region-out, select this many validation cities from each "
            "non-test region. Default: 1."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root

    input_csv: Path = args.input_csv or (
        instance_root
        / "metadata"
        / "instance_C_patches"
        / f"patch_metadata_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    )

    output_root: Path = args.output_root or (
        instance_root
        / "metadata"
        / "instance_C_splits"
    )

    log("STEP", "Building spatial CV splits for Instance C.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Input CSV:     {path_to_str(input_csv)}")
    log("INFO", f"Output root:   {path_to_str(output_root)}")
    log("INFO", f"Split family:  {args.split_family}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    rows = read_csv_rows(input_csv)
    validate_required_columns(rows)
    validate_unique_patch_ids(rows)

    original_fieldnames = list(rows[0].keys())

    log("OK", f"Loaded patch metadata rows: {len(rows)}")

    dataset_summary = summarize_rows(rows)
    city_stats = summarize_by_city(rows)
    region_stats = summarize_by_region(rows)

    cities = sorted(city_stats.keys())
    regions = sorted(region_stats.keys())

    log("INFO", f"Cities: {len(cities)}")
    log("INFO", f"Regions: {len(regions)}")
    log("INFO", f"Dataset positive patch percent: {dataset_summary['positive_patch_percent']:.6f}%")

    city_patch_counts = [int(city_stats[city]["patches"]) for city in cities]
    target_patch_count = float(median(city_patch_counts)) if city_patch_counts else 0.0
    target_positive_percent = float(dataset_summary["positive_patch_percent"])

    log("INFO", f"Validation target patch count: {target_patch_count:.2f}")
    log("INFO", f"Validation target positive percent: {target_positive_percent:.6f}%")

    output_root.mkdir(parents=True, exist_ok=True)

    if args.split_family in {"both", "leave_one_city_out"}:
        log("STEP", "Building leave-one-city-out folds.")

        loco_folds = build_leave_one_city_out_folds(
            rows=rows,
            city_stats=city_stats,
            target_positive_percent=target_positive_percent,
            target_patch_count=target_patch_count,
            min_val_patches=args.min_val_patches,
            min_val_positive_patches=args.min_val_positive_patches,
        )

        family_dir = output_root / "leave_one_city_out"

        write_family_outputs(
            folds=loco_folds,
            family_dir=family_dir,
            split_family="leave_one_city_out",
            original_fieldnames=original_fieldnames,
            overwrite=args.overwrite,
        )

        log("OK", f"Wrote leave-one-city-out folds: {len(loco_folds)}")
        log("OK", f"Output: {path_to_str(family_dir)}")

    if args.split_family in {"both", "leave_one_region_out"}:
        log("STEP", "Building leave-one-region-out folds.")

        loro_folds = build_leave_one_region_out_folds(
            rows=rows,
            city_stats=city_stats,
            region_stats=region_stats,
            target_positive_percent=target_positive_percent,
            target_patch_count=target_patch_count,
            min_val_patches=args.min_val_patches,
            min_val_positive_patches=args.min_val_positive_patches,
            loro_val_cities_per_region=args.loro_val_cities_per_region,
        )

        family_dir = output_root / "leave_one_region_out"

        write_family_outputs(
            folds=loro_folds,
            family_dir=family_dir,
            split_family="leave_one_region_out",
            original_fieldnames=original_fieldnames,
            overwrite=args.overwrite,
        )

        log("OK", f"Wrote leave-one-region-out folds: {len(loro_folds)}")
        log("OK", f"Output: {path_to_str(family_dir)}")

    log("STEP", "Done.")
    log("OK", f"Spatial split outputs written under: {path_to_str(output_root)}")


if __name__ == "__main__":
    main()