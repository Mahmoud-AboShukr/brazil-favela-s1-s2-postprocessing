#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_region_balanced_city_splits_224.py

Main objective
--------------
Build a simple region-balanced city-level train/validation/test split for the
next BigEarthNet/reBEN segmentation experiments.

The split design is:

    validation: one city from each Brazilian macro-region
    test:       one different city from each Brazilian macro-region
    train:      all remaining cities

Why this script exists
----------------------
The previous LORO experiments were intentionally strict and exposed strong
geographic domain shift. Supervisors suggested temporarily pausing LORO and
testing a simpler split first, to answer:

    Can a pretrained S1+S2 segmentation model learn favela masks at all
    when the training set contains examples from all regions?

This script creates the split CSV files needed for that experiment.

Expected input
--------------
By default, the script reads:

    <instance-root>/metadata/croma_probing/croma_comparison_manifest_ps224_st112_cover.csv

and filters it to:

    modality == s2_s1_snap_vv_vh

This gives one row per unique patch with:
    - optical_path
    - sar_path
    - label_path
    - city
    - region
    - row_start
    - col_start
    - label statistics

Expected output
---------------
By default, outputs are written to:

    <instance-root>/metadata/big_earth_net/region_balanced_city_split_ps224_st112_cover/

with:

    city_level_stats.csv
    selected_cities.json
    selected_city_summary.csv
    train.csv
    val.csv
    test.csv
    split_patch_summary.csv
    split_city_summary.csv
    split_report.md

Recommended command
-------------------
python src\\big_earth_net\\build_region_balanced_city_splits_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --target-modality "s2_s1_snap_vv_vh" `
  --patch-size 224 `
  --selection-strategy highest_positive `
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


REGION_ORDER = ["Central-West", "North", "Northeast", "South", "Southeast"]


# ---------------------------------------------------------------------
# Logging and utility helpers
# ---------------------------------------------------------------------

def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def warn(message: str) -> None:
    log("WARNING", message)


def fail(message: str, exit_code: int = 1) -> None:
    log("ERROR", message)
    raise SystemExit(exit_code)


def banner(title: str) -> None:
    print("[" + "=" * 100 + "]", flush=True)
    log("STEP", title)
    print("[" + "=" * 100 + "]", flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_to_str(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return str(path).replace("\\", "/")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        fail(
            "Output directory already exists and is not empty:\n"
            f"{path_to_str(path)}\n\n"
            "Use --overwrite if you want to replace/update the split outputs."
        )
    path.mkdir(parents=True, exist_ok=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return path_to_str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)


def normalize_city_name(name: Any) -> str:
    return str(name).strip()


def normalize_region_name(name: Any) -> str:
    return str(name).strip()


def split_manual_city_list(value: Optional[str]) -> List[str]:
    """
    Accept city lists separated by semicolon, comma, or pipe.

    Example:
        --val-cities "belem;recife;brasilia;porto_alegre;rio_de_janeiro"
    """
    if value is None:
        return []

    text = str(value).strip()
    if not text:
        return []

    parts = re.split(r"[;,|]+", text)
    return [normalize_city_name(p) for p in parts if normalize_city_name(p)]


def default_manifest_path(instance_root: Path) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / "croma_comparison_manifest_ps224_st112_cover.csv"
    )


def default_output_dir(instance_root: Path) -> Path:
    return (
        instance_root
        / "metadata"
        / "big_earth_net"
        / "region_balanced_city_split_ps224_st112_cover"
    )


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_manifest(
    manifest_path: Path,
    target_modality: str,
    allow_duplicate_patch_ids: bool,
) -> pd.DataFrame:
    if not manifest_path.exists():
        fail(f"Manifest file does not exist:\n{path_to_str(manifest_path)}")

    log("INFO", f"Reading manifest:\n{path_to_str(manifest_path)}")
    df = pd.read_csv(manifest_path)

    if df.empty:
        fail("Manifest is empty.")

    required = [
        "patch_id",
        "modality",
        "city",
        "region",
        "row_start",
        "col_start",
        "label_path",
        "label_binary",
        "label_positive_pixels",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(
            f"Manifest is missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    log("INFO", f"Manifest rows before modality filtering: {len(df):,}")

    modality_values = sorted(df["modality"].dropna().astype(str).unique().tolist())
    log("INFO", f"Available modalities: {modality_values}")

    df = df[df["modality"].astype(str) == str(target_modality)].copy()

    if df.empty:
        fail(
            f"No rows found for target modality: {target_modality}\n"
            f"Available modalities: {modality_values}"
        )

    log("INFO", f"Rows after modality filtering: {len(df):,}")

    df["city"] = df["city"].map(normalize_city_name)
    df["region"] = df["region"].map(normalize_region_name)

    df["label_binary"] = pd.to_numeric(df["label_binary"], errors="coerce").fillna(0).astype(int)
    df["label_positive_pixels"] = (
        pd.to_numeric(df["label_positive_pixels"], errors="coerce")
        .fillna(0)
        .astype(float)
    )

    df["row_start"] = pd.to_numeric(df["row_start"], errors="coerce").fillna(0).astype(int)
    df["col_start"] = pd.to_numeric(df["col_start"], errors="coerce").fillna(0).astype(int)

    duplicate_patch_count = int(df["patch_id"].duplicated().sum())

    if duplicate_patch_count > 0:
        message = (
            f"Found {duplicate_patch_count:,} duplicated patch_id values after modality filtering."
        )

        if allow_duplicate_patch_ids:
            warn(message)
            warn("Keeping the first row for each duplicated patch_id.")
            df = df.drop_duplicates(subset=["patch_id"], keep="first").copy()
        else:
            duplicated = (
                df[df["patch_id"].duplicated(keep=False)]
                .sort_values("patch_id")
                .head(20)
            )
            fail(
                message
                + "\n\nFirst duplicated examples:\n"
                + duplicated[["patch_id", "city", "region", "modality"]].to_string(index=False)
                + "\n\nUse --allow-duplicate-patch-ids if you intentionally want to keep first occurrence."
            )

    if "optical_path" not in df.columns:
        warn("Column optical_path is missing. Training script may need this later.")

    if "sar_path" not in df.columns:
        warn("Column sar_path is missing. Training script may need this later.")

    unknown_regions = sorted([r for r in df["region"].unique().tolist() if r not in REGION_ORDER])
    if unknown_regions:
        warn(
            "Found regions outside expected REGION_ORDER:\n"
            f"{unknown_regions}\n"
            "They will still be included, but auto-selection is designed for the five standard regions."
        )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# City statistics
# ---------------------------------------------------------------------

def build_city_level_stats(df: pd.DataFrame, patch_size: int) -> pd.DataFrame:
    patch_area = int(patch_size) * int(patch_size)

    stats = (
        df.groupby(["region", "city"], dropna=False)
        .agg(
            n_patches=("patch_id", "count"),
            n_positive_patches=("label_binary", "sum"),
            label_positive_pixels=("label_positive_pixels", "sum"),
            mean_patch_label_positive_pixels=("label_positive_pixels", "mean"),
            max_patch_label_positive_pixels=("label_positive_pixels", "max"),
        )
        .reset_index()
    )

    stats["n_empty_patches"] = stats["n_patches"] - stats["n_positive_patches"]
    stats["positive_patch_pct"] = (
        100.0 * stats["n_positive_patches"] / stats["n_patches"].clip(lower=1)
    )

    stats["total_pixels"] = stats["n_patches"] * patch_area
    stats["label_positive_percent"] = (
        100.0 * stats["label_positive_pixels"] / stats["total_pixels"].clip(lower=1)
    )

    stats["has_positive_pixels"] = stats["label_positive_pixels"] > 0
    stats["has_positive_patches"] = stats["n_positive_patches"] > 0

    region_rank = {region: i for i, region in enumerate(REGION_ORDER)}
    stats["_region_order"] = stats["region"].map(region_rank).fillna(999).astype(int)

    stats = stats.sort_values(
        ["_region_order", "region", "label_positive_pixels", "n_positive_patches", "n_patches"],
        ascending=[True, True, False, False, False],
    ).drop(columns=["_region_order"])

    return stats.reset_index(drop=True)


def print_city_stats_summary(city_stats: pd.DataFrame) -> None:
    log("INFO", "City counts by region:")

    region_summary = (
        city_stats.groupby("region")
        .agg(
            n_cities=("city", "count"),
            n_patches=("n_patches", "sum"),
            n_positive_patches=("n_positive_patches", "sum"),
            label_positive_pixels=("label_positive_pixels", "sum"),
        )
        .reset_index()
    )

    for _, row in region_summary.iterrows():
        log(
            "INFO",
            f"  {row['region']}: "
            f"{int(row['n_cities'])} cities, "
            f"{int(row['n_patches']):,} patches, "
            f"{int(row['n_positive_patches']):,} positive patches"
        )


# ---------------------------------------------------------------------
# City selection
# ---------------------------------------------------------------------

def validate_manual_cities(
    city_stats: pd.DataFrame,
    val_cities: Sequence[str],
    test_cities: Sequence[str],
    expected_regions: Sequence[str],
) -> Tuple[List[str], List[str]]:
    val_set = {normalize_city_name(c) for c in val_cities}
    test_set = {normalize_city_name(c) for c in test_cities}

    if not val_set or not test_set:
        fail("Both --val-cities and --test-cities must be provided for manual selection.")

    overlap = val_set.intersection(test_set)
    if overlap:
        fail(f"Manual val/test city lists overlap: {sorted(overlap)}")

    known_cities = set(city_stats["city"].tolist())

    unknown_val = sorted(val_set - known_cities)
    unknown_test = sorted(test_set - known_cities)

    if unknown_val:
        fail(f"Unknown validation cities: {unknown_val}")

    if unknown_test:
        fail(f"Unknown test cities: {unknown_test}")

    city_to_region = dict(zip(city_stats["city"], city_stats["region"]))

    val_regions = [city_to_region[c] for c in val_set]
    test_regions = [city_to_region[c] for c in test_set]

    missing_val_regions = sorted(set(expected_regions) - set(val_regions))
    missing_test_regions = sorted(set(expected_regions) - set(test_regions))

    duplicate_val_regions = sorted([r for r in set(val_regions) if val_regions.count(r) > 1])
    duplicate_test_regions = sorted([r for r in set(test_regions) if test_regions.count(r) > 1])

    if missing_val_regions:
        fail(f"Manual validation cities do not cover regions: {missing_val_regions}")

    if missing_test_regions:
        fail(f"Manual test cities do not cover regions: {missing_test_regions}")

    if duplicate_val_regions:
        fail(f"Manual validation has more than one city for regions: {duplicate_val_regions}")

    if duplicate_test_regions:
        fail(f"Manual test has more than one city for regions: {duplicate_test_regions}")

    return sorted(val_set), sorted(test_set)


def eligible_cities_for_region(
    city_stats: pd.DataFrame,
    region: str,
    min_patches: int,
    min_positive_patches: int,
    min_positive_pixels: float,
) -> pd.DataFrame:
    region_df = city_stats[city_stats["region"] == region].copy()

    if region_df.empty:
        return region_df

    eligible = region_df[
        (region_df["n_patches"] >= int(min_patches))
        & (region_df["n_positive_patches"] >= int(min_positive_patches))
        & (region_df["label_positive_pixels"] >= float(min_positive_pixels))
    ].copy()

    return eligible


def auto_select_cities(
    city_stats: pd.DataFrame,
    expected_regions: Sequence[str],
    min_patches: int,
    min_positive_patches: int,
    min_positive_pixels: float,
    selection_strategy: str,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Select one validation city and one test city per region.

    Strategies:
      highest_positive:
        Sort cities by label_positive_pixels descending.
        Test gets the highest-positive city.
        Val gets the second-highest-positive city.

      balanced_positive_percent:
        Sort eligible cities by label_positive_percent.
        Test gets the city near the upper-middle.
        Val gets the city near the middle.
        This avoids always selecting only the most positive cities.

      largest_patch_count:
        Sort by n_patches descending.
        Test gets largest city, val gets second largest.
    """
    selection_strategy = str(selection_strategy)

    val_cities: List[str] = []
    test_cities: List[str] = []
    details: Dict[str, Any] = {
        "selection_strategy": selection_strategy,
        "regions": {},
        "warnings": [],
    }

    for region in expected_regions:
        region_df = city_stats[city_stats["region"] == region].copy()

        if region_df.empty:
            fail(f"No cities found for region: {region}")

        eligible = eligible_cities_for_region(
            city_stats=city_stats,
            region=region,
            min_patches=min_patches,
            min_positive_patches=min_positive_patches,
            min_positive_pixels=min_positive_pixels,
        )

        if len(eligible) < 2:
            warning = (
                f"Region {region} has only {len(eligible)} eligible cities using strict filters. "
                "Relaxing filters for this region."
            )
            warn(warning)
            details["warnings"].append(warning)

            eligible = region_df[
                (region_df["n_positive_patches"] > 0)
                | (region_df["label_positive_pixels"] > 0)
            ].copy()

        if len(eligible) < 2:
            warning = (
                f"Region {region} still has fewer than two positive cities. "
                "Using all cities in the region."
            )
            warn(warning)
            details["warnings"].append(warning)
            eligible = region_df.copy()

        if len(eligible) < 2:
            fail(
                f"Region {region} has fewer than two cities. "
                "Cannot select one validation and one test city."
            )

        if selection_strategy == "highest_positive":
            ranked = eligible.sort_values(
                ["label_positive_pixels", "n_positive_patches", "n_patches", "city"],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)

            test_city = str(ranked.iloc[0]["city"])
            val_city = str(ranked.iloc[1]["city"])

        elif selection_strategy == "balanced_positive_percent":
            ranked = eligible.sort_values(
                ["label_positive_percent", "n_positive_patches", "n_patches", "city"],
                ascending=[True, True, True, True],
            ).reset_index(drop=True)

            n = len(ranked)
            val_idx = max(0, min(n - 1, int(round(0.50 * (n - 1)))))
            test_idx = max(0, min(n - 1, int(round(0.75 * (n - 1)))))

            if test_idx == val_idx:
                test_idx = min(n - 1, val_idx + 1)

            val_city = str(ranked.iloc[val_idx]["city"])
            test_city = str(ranked.iloc[test_idx]["city"])

        elif selection_strategy == "largest_patch_count":
            ranked = eligible.sort_values(
                ["n_patches", "n_positive_patches", "label_positive_pixels", "city"],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)

            test_city = str(ranked.iloc[0]["city"])
            val_city = str(ranked.iloc[1]["city"])

        else:
            fail(
                f"Unsupported selection strategy: {selection_strategy}. "
                "Use highest_positive, balanced_positive_percent, or largest_patch_count."
            )

        if val_city == test_city:
            fail(f"Auto-selection selected the same city for val/test in region {region}: {val_city}")

        val_cities.append(val_city)
        test_cities.append(test_city)

        details["regions"][region] = {
            "n_region_cities": int(len(region_df)),
            "n_eligible_cities": int(len(eligible)),
            "selected_val_city": val_city,
            "selected_test_city": test_city,
            "eligible_cities_ranked": ranked[
                [
                    "city",
                    "n_patches",
                    "n_positive_patches",
                    "positive_patch_pct",
                    "label_positive_pixels",
                    "label_positive_percent",
                ]
            ].to_dict(orient="records"),
        }

    return val_cities, test_cities, details


# ---------------------------------------------------------------------
# Split creation and validation
# ---------------------------------------------------------------------

def assign_splits(
    df: pd.DataFrame,
    val_cities: Sequence[str],
    test_cities: Sequence[str],
) -> pd.DataFrame:
    val_set = {normalize_city_name(c) for c in val_cities}
    test_set = {normalize_city_name(c) for c in test_cities}

    overlap = val_set.intersection(test_set)
    if overlap:
        fail(f"Validation and test cities overlap: {sorted(overlap)}")

    out = df.copy()
    out["split"] = "train"
    out.loc[out["city"].isin(val_set), "split"] = "val"
    out.loc[out["city"].isin(test_set), "split"] = "test"

    return out


def validate_split(df: pd.DataFrame) -> Dict[str, Any]:
    split_names = ["train", "val", "test"]

    split_city_sets = {
        split: set(df[df["split"] == split]["city"].unique().tolist())
        for split in split_names
    }

    split_patch_sets = {
        split: set(df[df["split"] == split]["patch_id"].unique().tolist())
        for split in split_names
    }

    problems: List[str] = []

    for a in split_names:
        for b in split_names:
            if a >= b:
                continue

            city_overlap = split_city_sets[a].intersection(split_city_sets[b])
            patch_overlap = split_patch_sets[a].intersection(split_patch_sets[b])

            if city_overlap:
                problems.append(f"City leakage between {a} and {b}: {sorted(city_overlap)}")

            if patch_overlap:
                examples = sorted(list(patch_overlap))[:10]
                problems.append(
                    f"Patch leakage between {a} and {b}: {len(patch_overlap)} examples. "
                    f"First examples: {examples}"
                )

    split_counts = df["split"].value_counts().to_dict()

    for split in split_names:
        if split_counts.get(split, 0) == 0:
            problems.append(f"Split {split} is empty.")

    if problems:
        fail("Split validation failed:\n" + "\n".join(f"- {p}" for p in problems))

    return {
        "status": "ok",
        "split_patch_counts": {k: int(v) for k, v in split_counts.items()},
        "split_city_counts": {
            split: int(len(cities))
            for split, cities in split_city_sets.items()
        },
        "city_overlap": "none",
        "patch_overlap": "none",
    }


def build_split_patch_summary(df: pd.DataFrame, patch_size: int) -> pd.DataFrame:
    patch_area = int(patch_size) * int(patch_size)

    summary = (
        df.groupby("split")
        .agg(
            n_patches=("patch_id", "count"),
            n_cities=("city", "nunique"),
            n_regions=("region", "nunique"),
            n_positive_patches=("label_binary", "sum"),
            label_positive_pixels=("label_positive_pixels", "sum"),
        )
        .reset_index()
    )

    summary["n_empty_patches"] = summary["n_patches"] - summary["n_positive_patches"]
    summary["positive_patch_pct"] = (
        100.0 * summary["n_positive_patches"] / summary["n_patches"].clip(lower=1)
    )
    summary["total_pixels"] = summary["n_patches"] * patch_area
    summary["label_positive_percent"] = (
        100.0 * summary["label_positive_pixels"] / summary["total_pixels"].clip(lower=1)
    )

    split_order = {"train": 0, "val": 1, "test": 2}
    summary["_order"] = summary["split"].map(split_order).fillna(99).astype(int)
    summary = summary.sort_values("_order").drop(columns=["_order"])

    return summary.reset_index(drop=True)


def build_split_city_summary(df: pd.DataFrame, patch_size: int) -> pd.DataFrame:
    patch_area = int(patch_size) * int(patch_size)

    summary = (
        df.groupby(["split", "region", "city"])
        .agg(
            n_patches=("patch_id", "count"),
            n_positive_patches=("label_binary", "sum"),
            label_positive_pixels=("label_positive_pixels", "sum"),
        )
        .reset_index()
    )

    summary["positive_patch_pct"] = (
        100.0 * summary["n_positive_patches"] / summary["n_patches"].clip(lower=1)
    )
    summary["total_pixels"] = summary["n_patches"] * patch_area
    summary["label_positive_percent"] = (
        100.0 * summary["label_positive_pixels"] / summary["total_pixels"].clip(lower=1)
    )

    split_order = {"train": 0, "val": 1, "test": 2}
    region_order = {region: i for i, region in enumerate(REGION_ORDER)}

    summary["_split_order"] = summary["split"].map(split_order).fillna(99).astype(int)
    summary["_region_order"] = summary["region"].map(region_order).fillna(999).astype(int)

    summary = summary.sort_values(
        ["_split_order", "_region_order", "region", "city"]
    ).drop(columns=["_split_order", "_region_order"])

    return summary.reset_index(drop=True)


def build_selected_city_summary(
    city_stats: pd.DataFrame,
    val_cities: Sequence[str],
    test_cities: Sequence[str],
) -> pd.DataFrame:
    val_set = set(val_cities)
    test_set = set(test_cities)

    selected = city_stats[city_stats["city"].isin(val_set.union(test_set))].copy()

    selected["selected_split"] = "unknown"
    selected.loc[selected["city"].isin(val_set), "selected_split"] = "val"
    selected.loc[selected["city"].isin(test_set), "selected_split"] = "test"

    split_order = {"val": 0, "test": 1}
    region_order = {region: i for i, region in enumerate(REGION_ORDER)}

    selected["_split_order"] = selected["selected_split"].map(split_order).fillna(99).astype(int)
    selected["_region_order"] = selected["region"].map(region_order).fillna(999).astype(int)

    selected = selected.sort_values(
        ["_region_order", "selected_split", "city"]
    ).drop(columns=["_split_order", "_region_order"])

    return selected.reset_index(drop=True)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def markdown_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows).copy()

    if df.empty:
        return "_No rows._"

    return df.to_markdown(index=False)


def write_split_report(
    path: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    output_dir: Path,
    city_stats: pd.DataFrame,
    selected_city_summary: pd.DataFrame,
    split_patch_summary: pd.DataFrame,
    split_city_summary: pd.DataFrame,
    validation: Dict[str, Any],
    selection_details: Dict[str, Any],
) -> None:
    lines: List[str] = []

    lines.append("# Region-Balanced City Split Report")
    lines.append("")
    lines.append(f"Created UTC: `{now_utc()}`")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Create a simpler city-level split for the BigEarthNet/reBEN segmentation experiments. "
        "Validation and test each contain one city from every Brazilian macro-region, while "
        "the training split contains all remaining cities."
    )
    lines.append("")
    lines.append("This split is less strict than Leave-One-Region-Out because all regions remain represented in training.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Instance root: `{args.instance_root}`")
    lines.append(f"- Manifest path: `{path_to_str(manifest_path)}`")
    lines.append(f"- Output directory: `{path_to_str(output_dir)}`")
    lines.append(f"- Target modality: `{args.target_modality}`")
    lines.append(f"- Patch size: `{args.patch_size}`")
    lines.append(f"- Selection strategy: `{args.selection_strategy}`")
    lines.append(f"- Minimum patches per selected city: `{args.min_patches}`")
    lines.append(f"- Minimum positive patches per selected city: `{args.min_positive_patches}`")
    lines.append(f"- Minimum positive pixels per selected city: `{args.min_positive_pixels}`")
    lines.append("")
    lines.append("## Selected Cities")
    lines.append("")
    lines.append(markdown_table(selected_city_summary))
    lines.append("")
    lines.append("## Split-Level Summary")
    lines.append("")
    lines.append(markdown_table(split_patch_summary))
    lines.append("")
    lines.append("## City-Level Split Summary")
    lines.append("")
    lines.append(markdown_table(split_city_summary))
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(jsonable(validation), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Selection Details")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(jsonable(selection_details), indent=2, ensure_ascii=False)[:12000])
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This split is intended for learnability testing. If a BigEarthNet/reBEN-pretrained "
        "segmentation model performs well under this split, but poorly under LORO, then the "
        "dataset likely contains learnable signal and the LORO difficulty is mainly due to "
        "geographic domain shift. If performance is poor even under this easier split, then "
        "the issue may be deeper, involving labels, normalization, class imbalance, alignment, "
        "model architecture, or input modality handling."
    )

    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    instance_root = Path(args.instance_root)
    manifest_path = Path(args.manifest_path) if args.manifest_path else default_manifest_path(instance_root)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(instance_root)

    banner("Build region-balanced city split for BigEarthNet/reBEN experiments")

    log("INFO", f"Instance root:      {path_to_str(instance_root)}")
    log("INFO", f"Manifest path:      {path_to_str(manifest_path)}")
    log("INFO", f"Output directory:   {path_to_str(output_dir)}")
    log("INFO", f"Target modality:    {args.target_modality}")
    log("INFO", f"Selection strategy: {args.selection_strategy}")

    ensure_output_dir(output_dir, overwrite=bool(args.overwrite))

    df = load_manifest(
        manifest_path=manifest_path,
        target_modality=str(args.target_modality),
        allow_duplicate_patch_ids=bool(args.allow_duplicate_patch_ids),
    )

    city_stats = build_city_level_stats(df, patch_size=int(args.patch_size))
    print_city_stats_summary(city_stats)

    city_stats_path = output_dir / "city_level_stats.csv"
    city_stats.to_csv(city_stats_path, index=False)

    expected_regions = REGION_ORDER

    manual_val_cities = split_manual_city_list(args.val_cities)
    manual_test_cities = split_manual_city_list(args.test_cities)

    if manual_val_cities or manual_test_cities:
        log("INFO", "Using manual city selection.")
        val_cities, test_cities = validate_manual_cities(
            city_stats=city_stats,
            val_cities=manual_val_cities,
            test_cities=manual_test_cities,
            expected_regions=expected_regions,
        )

        selection_details = {
            "mode": "manual",
            "val_cities": val_cities,
            "test_cities": test_cities,
        }

    else:
        log("INFO", "Using automatic city selection.")
        val_cities, test_cities, selection_details = auto_select_cities(
            city_stats=city_stats,
            expected_regions=expected_regions,
            min_patches=int(args.min_patches),
            min_positive_patches=int(args.min_positive_patches),
            min_positive_pixels=float(args.min_positive_pixels),
            selection_strategy=str(args.selection_strategy),
        )
        selection_details["mode"] = "automatic"

    log("INFO", f"Selected validation cities: {val_cities}")
    log("INFO", f"Selected test cities:       {test_cities}")

    split_df = assign_splits(
        df=df,
        val_cities=val_cities,
        test_cities=test_cities,
    )

    validation = validate_split(split_df)

    selected_city_summary = build_selected_city_summary(
        city_stats=city_stats,
        val_cities=val_cities,
        test_cities=test_cities,
    )

    split_patch_summary = build_split_patch_summary(
        df=split_df,
        patch_size=int(args.patch_size),
    )

    split_city_summary = build_split_city_summary(
        df=split_df,
        patch_size=int(args.patch_size),
    )

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    selected_city_summary_path = output_dir / "selected_city_summary.csv"
    split_patch_summary_path = output_dir / "split_patch_summary.csv"
    split_city_summary_path = output_dir / "split_city_summary.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    selected_city_summary.to_csv(selected_city_summary_path, index=False)
    split_patch_summary.to_csv(split_patch_summary_path, index=False)
    split_city_summary.to_csv(split_city_summary_path, index=False)

    selected_payload = {
        "created_utc": now_utc(),
        "instance_root": path_to_str(instance_root),
        "manifest_path": path_to_str(manifest_path),
        "output_dir": path_to_str(output_dir),
        "target_modality": str(args.target_modality),
        "patch_size": int(args.patch_size),
        "val_cities": val_cities,
        "test_cities": test_cities,
        "train_cities": sorted(train_df["city"].unique().tolist()),
        "selection_details": selection_details,
        "validation": validation,
        "outputs": {
            "city_level_stats_csv": path_to_str(city_stats_path),
            "selected_city_summary_csv": path_to_str(selected_city_summary_path),
            "split_patch_summary_csv": path_to_str(split_patch_summary_path),
            "split_city_summary_csv": path_to_str(split_city_summary_path),
            "train_csv": path_to_str(train_path),
            "val_csv": path_to_str(val_path),
            "test_csv": path_to_str(test_path),
        },
    }

    selected_json_path = output_dir / "selected_cities.json"
    write_json(selected_json_path, selected_payload)

    report_path = output_dir / "split_report.md"
    write_split_report(
        path=report_path,
        args=args,
        manifest_path=manifest_path,
        output_dir=output_dir,
        city_stats=city_stats,
        selected_city_summary=selected_city_summary,
        split_patch_summary=split_patch_summary,
        split_city_summary=split_city_summary,
        validation=validation,
        selection_details=selection_details,
    )

    banner("Completed")

    log("OK", f"City-level stats:          {path_to_str(city_stats_path)}")
    log("OK", f"Selected cities JSON:      {path_to_str(selected_json_path)}")
    log("OK", f"Selected city summary:     {path_to_str(selected_city_summary_path)}")
    log("OK", f"Split patch summary:       {path_to_str(split_patch_summary_path)}")
    log("OK", f"Split city summary:        {path_to_str(split_city_summary_path)}")
    log("OK", f"Train CSV:                 {path_to_str(train_path)}")
    log("OK", f"Val CSV:                   {path_to_str(val_path)}")
    log("OK", f"Test CSV:                  {path_to_str(test_path)}")
    log("OK", f"Report:                    {path_to_str(report_path)}")

    log("INFO", "Split-level summary:")
    print(split_patch_summary.to_string(index=False), flush=True)

    log("INFO", "Selected cities:")
    print(
        selected_city_summary[
            [
                "selected_split",
                "region",
                "city",
                "n_patches",
                "n_positive_patches",
                "positive_patch_pct",
                "label_positive_percent",
            ]
        ].to_string(index=False),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build region-balanced city train/val/test splits for BigEarthNet/reBEN favela segmentation experiments."
    )

    parser.add_argument(
        "--instance-root",
        required=True,
        help="Dataset instance root, e.g. D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional explicit manifest path. Default: instance-root/metadata/croma_probing/croma_comparison_manifest_ps224_st112_cover.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Default: instance-root/metadata/big_earth_net/region_balanced_city_split_ps224_st112_cover",
    )
    parser.add_argument(
        "--target-modality",
        default="s2_s1_snap_vv_vh",
        help="Manifest modality to use. Default: s2_s1_snap_vv_vh",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size used for positive pixel percentage calculations.",
    )

    parser.add_argument(
        "--selection-strategy",
        choices=["highest_positive", "balanced_positive_percent", "largest_patch_count"],
        default="highest_positive",
        help=(
            "Automatic city selection strategy. "
            "highest_positive selects the two cities with most positive pixels per region. "
            "balanced_positive_percent selects middle/upper density cities. "
            "largest_patch_count selects largest patch-count cities."
        ),
    )

    parser.add_argument(
        "--min-patches",
        type=int,
        default=10,
        help="Minimum number of patches for a city to be eligible for automatic val/test selection.",
    )
    parser.add_argument(
        "--min-positive-patches",
        type=int,
        default=1,
        help="Minimum number of positive patches for a city to be eligible for automatic val/test selection.",
    )
    parser.add_argument(
        "--min-positive-pixels",
        type=float,
        default=1.0,
        help="Minimum number of positive pixels for a city to be eligible for automatic val/test selection.",
    )

    parser.add_argument(
        "--val-cities",
        default=None,
        help=(
            "Manual validation city list separated by semicolon/comma/pipe. "
            "Must contain exactly one city per region if provided."
        ),
    )
    parser.add_argument(
        "--test-cities",
        default=None,
        help=(
            "Manual test city list separated by semicolon/comma/pipe. "
            "Must contain exactly one city per region if provided."
        ),
    )

    parser.add_argument(
        "--allow-duplicate-patch-ids",
        action="store_true",
        help="If duplicate patch_id values exist after modality filtering, keep the first occurrence instead of failing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())