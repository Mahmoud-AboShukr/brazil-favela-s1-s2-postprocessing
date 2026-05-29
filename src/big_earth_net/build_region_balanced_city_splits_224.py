#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_region_balanced_city_splits_224.py

Main objective
--------------
Build a simple region-balanced train/validation/test split for BigEarthNet/reBEN
segmentation experiments.

Updated split design
--------------------
The goal is to test learnability, not strict geographic generalization.

Therefore, the training split MUST contain examples from all Brazilian regions.

For regions with at least 3 cities:
    test  = one city
    val   = one different city
    train = all remaining cities

For regions with exactly 2 cities:
    test  = one city
    train = the other city
    val   = a patch-level subset sampled from the train city

This is a deliberate compromise. It keeps the test split city-held-out, while
ensuring the model sees all regions during training.

Expected output
---------------
<instance-root>/metadata/big_earth_net/region_balanced_city_split_ps224_st112_cover/

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
  --split-design train_region_covered `
  --city-ranking highest_positive `
  --val-patch-fraction 0.20 `
  --min-patch-val-patches 50 `
  --max-patch-val-patches 250 `
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
# Logging and utilities
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
            "Use --overwrite if you want to update the split outputs."
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


def safe_markdown_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows).copy()

    if df.empty:
        return "_No rows._"

    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


# ---------------------------------------------------------------------
# Data loading
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
    df["label_positive_pixels"] = pd.to_numeric(
        df["label_positive_pixels"],
        errors="coerce",
    ).fillna(0).astype(float)

    df["row_start"] = pd.to_numeric(df["row_start"], errors="coerce").fillna(0).astype(int)
    df["col_start"] = pd.to_numeric(df["col_start"], errors="coerce").fillna(0).astype(int)

    duplicate_patch_count = int(df["patch_id"].duplicated().sum())

    if duplicate_patch_count > 0:
        message = f"Found {duplicate_patch_count:,} duplicated patch_id values after modality filtering."

        if allow_duplicate_patch_ids:
            warn(message)
            warn("Keeping first occurrence per patch_id.")
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
                + "\n\nUse --allow-duplicate-patch-ids only if this is intentional."
            )

    if "optical_path" not in df.columns:
        warn("Column optical_path is missing. Training script may need this later.")

    if "sar_path" not in df.columns:
        warn("Column sar_path is missing. Training script may need this later.")

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
    stats["positive_patch_pct"] = 100.0 * stats["n_positive_patches"] / stats["n_patches"].clip(lower=1)
    stats["total_pixels"] = stats["n_patches"] * patch_area
    stats["label_positive_percent"] = (
        100.0 * stats["label_positive_pixels"] / stats["total_pixels"].clip(lower=1)
    )
    stats["has_positive_pixels"] = stats["label_positive_pixels"] > 0
    stats["has_positive_patches"] = stats["n_positive_patches"] > 0

    region_rank = {region: i for i, region in enumerate(REGION_ORDER)}
    stats["_region_order"] = stats["region"].map(region_rank).fillna(999).astype(int)

    stats = (
        stats.sort_values(
            ["_region_order", "region", "label_positive_pixels", "n_positive_patches", "n_patches"],
            ascending=[True, True, False, False, False],
        )
        .drop(columns=["_region_order"])
        .reset_index(drop=True)
    )

    return stats


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
            f"{int(row['n_positive_patches']):,} positive patches, "
            f"{int(row['label_positive_pixels']):,} positive pixels"
        )


def rank_cities(region_df: pd.DataFrame, city_ranking: str) -> pd.DataFrame:
    if city_ranking == "highest_positive":
        return region_df.sort_values(
            ["label_positive_pixels", "n_positive_patches", "n_patches", "city"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    if city_ranking == "balanced_positive_percent":
        return region_df.sort_values(
            ["label_positive_percent", "n_positive_patches", "n_patches", "city"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    if city_ranking == "largest_patch_count":
        return region_df.sort_values(
            ["n_patches", "n_positive_patches", "label_positive_pixels", "city"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    fail(f"Unsupported city ranking: {city_ranking}")


# ---------------------------------------------------------------------
# Train-region-covered split logic
# ---------------------------------------------------------------------

def stratified_patch_sample(
    city_df: pd.DataFrame,
    val_fraction: float,
    min_val_patches: int,
    max_val_patches: int,
    seed: int,
) -> Set[str]:
    """
    Sample validation patches from a train city while preserving approximate
    positive/empty ratio.

    This is used only for regions with exactly 2 cities, where we cannot have
    one train city, one validation city, and one test city simultaneously.
    """
    n = len(city_df)

    if n < 2:
        fail("Cannot sample validation patches from a city with fewer than 2 patches.")

    desired = int(round(float(val_fraction) * n))
    desired = max(int(min_val_patches), desired)
    desired = min(int(max_val_patches), desired)
    desired = min(desired, n - 1)

    if desired <= 0:
        desired = 1

    pos_df = city_df[city_df["label_binary"] == 1]
    neg_df = city_df[city_df["label_binary"] == 0]

    pos_count = len(pos_df)
    neg_count = len(neg_df)

    if pos_count == 0 or neg_count == 0:
        sampled = city_df.sample(n=desired, random_state=seed)
        return set(sampled["patch_id"].astype(str).tolist())

    pos_ratio = pos_count / n
    n_pos_val = int(round(desired * pos_ratio))
    n_pos_val = max(1, n_pos_val)
    n_pos_val = min(pos_count, n_pos_val)

    n_neg_val = desired - n_pos_val
    n_neg_val = max(0, n_neg_val)
    n_neg_val = min(neg_count, n_neg_val)

    # If rounding/capping reduced the total, fill from the remaining pool.
    selected_parts = []

    selected_pos_ids: Set[str] = set()
    selected_neg_ids: Set[str] = set()

    if n_pos_val > 0:
        pos_sample = pos_df.sample(n=n_pos_val, random_state=seed)
        selected_parts.append(pos_sample)
        selected_pos_ids = set(pos_sample["patch_id"].astype(str).tolist())

    if n_neg_val > 0:
        neg_sample = neg_df.sample(n=n_neg_val, random_state=seed + 1)
        selected_parts.append(neg_sample)
        selected_neg_ids = set(neg_sample["patch_id"].astype(str).tolist())

    selected = pd.concat(selected_parts, axis=0) if selected_parts else pd.DataFrame()

    if len(selected) < desired:
        already = selected_pos_ids.union(selected_neg_ids)
        remaining = city_df[~city_df["patch_id"].astype(str).isin(already)]
        fill_n = min(desired - len(selected), len(remaining))

        if fill_n > 0:
            fill_sample = remaining.sample(n=fill_n, random_state=seed + 2)
            selected = pd.concat([selected, fill_sample], axis=0)

    return set(selected["patch_id"].astype(str).tolist())


def build_train_region_covered_split(
    df: pd.DataFrame,
    city_stats: pd.DataFrame,
    city_ranking: str,
    val_patch_fraction: float,
    min_patch_val_patches: int,
    max_patch_val_patches: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build split such that train contains all regions.

    For >=3 cities:
        test city = top-ranked city
        val city  = second-ranked city
        train     = rest

    For exactly 2 cities:
        test city = top-ranked city
        train city = second-ranked city
        val patches = sampled from train city

    This creates city-level test everywhere, city-level validation where possible,
    and patch-level validation only where necessary.
    """
    split_df = df.copy()
    split_df["split"] = "train"
    split_df["split_assignment_type"] = "train_city"
    split_df["split_note"] = ""

    selected: Dict[str, Any] = {
        "split_design": "train_region_covered",
        "city_ranking": city_ranking,
        "val_patch_fraction": float(val_patch_fraction),
        "min_patch_val_patches": int(min_patch_val_patches),
        "max_patch_val_patches": int(max_patch_val_patches),
        "seed": int(seed),
        "regions": {},
        "test_cities": [],
        "val_city_holdout_cities": [],
        "patch_val_source_cities": [],
        "warnings": [],
    }

    for region in REGION_ORDER:
        region_city_stats = city_stats[city_stats["region"] == region].copy()

        if region_city_stats.empty:
            fail(f"No cities found for expected region: {region}")

        ranked = rank_cities(region_city_stats, city_ranking=city_ranking)
        n_cities = len(ranked)

        if n_cities < 2:
            fail(
                f"Region {region} has only {n_cities} city. "
                "Cannot create a city-held-out test and train-region-covered split."
            )

        # For highest_positive/largest_patch_count, top row is strongest/largest.
        # For balanced_positive_percent, rank_cities returns low-to-high.
        # For balanced mode, choose test from upper-middle and val/train from middle.
        if city_ranking == "balanced_positive_percent" and n_cities >= 3:
            test_idx = max(0, min(n_cities - 1, int(round(0.75 * (n_cities - 1)))))
            val_idx = max(0, min(n_cities - 1, int(round(0.50 * (n_cities - 1)))))

            if test_idx == val_idx:
                test_idx = min(n_cities - 1, val_idx + 1)

            test_city = str(ranked.iloc[test_idx]["city"])
            val_city = str(ranked.iloc[val_idx]["city"])

        else:
            test_city = str(ranked.iloc[0]["city"])
            val_city = str(ranked.iloc[1]["city"])

        selected["test_cities"].append(test_city)

        split_df.loc[split_df["city"] == test_city, "split"] = "test"
        split_df.loc[split_df["city"] == test_city, "split_assignment_type"] = "test_city_holdout"
        split_df.loc[split_df["city"] == test_city, "split_note"] = f"{region}: city-held-out test"

        if n_cities >= 3:
            selected["val_city_holdout_cities"].append(val_city)

            split_df.loc[split_df["city"] == val_city, "split"] = "val"
            split_df.loc[split_df["city"] == val_city, "split_assignment_type"] = "val_city_holdout"
            split_df.loc[split_df["city"] == val_city, "split_note"] = f"{region}: city-held-out validation"

            train_cities = [
                str(c)
                for c in ranked["city"].tolist()
                if str(c) not in {test_city, val_city}
            ]

            selected["regions"][region] = {
                "n_cities": int(n_cities),
                "mode": "city_val_and_city_test",
                "test_city": test_city,
                "val_city": val_city,
                "train_cities": train_cities,
                "ranked_cities": ranked[
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

        else:
            # Exactly 2 cities: keep second city in train, sample validation patches from it.
            train_val_city = val_city
            selected["patch_val_source_cities"].append(train_val_city)

            source_df = split_df[
                (split_df["region"] == region)
                & (split_df["city"] == train_val_city)
                & (split_df["split"] == "train")
            ].copy()

            val_patch_ids = stratified_patch_sample(
                city_df=source_df,
                val_fraction=float(val_patch_fraction),
                min_val_patches=int(min_patch_val_patches),
                max_val_patches=int(max_patch_val_patches),
                seed=int(seed) + len(selected["patch_val_source_cities"]) * 100,
            )

            patch_mask = split_df["patch_id"].astype(str).isin(val_patch_ids)

            split_df.loc[patch_mask, "split"] = "val"
            split_df.loc[patch_mask, "split_assignment_type"] = "val_patch_sample_from_train_city"
            split_df.loc[patch_mask, "split_note"] = (
                f"{region}: patch-level validation sampled from train city {train_val_city} "
                "to preserve train region coverage"
            )

            train_patch_count_after = int(
                len(
                    split_df[
                        (split_df["region"] == region)
                        & (split_df["city"] == train_val_city)
                        & (split_df["split"] == "train")
                    ]
                )
            )

            selected["regions"][region] = {
                "n_cities": int(n_cities),
                "mode": "city_test_patch_val_train_region_covered",
                "test_city": test_city,
                "patch_val_source_city": train_val_city,
                "patch_val_count": int(len(val_patch_ids)),
                "train_patch_count_remaining_in_source_city": train_patch_count_after,
                "ranked_cities": ranked[
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

            warning = (
                f"Region {region} has only 2 cities. "
                f"Using {test_city} as city-held-out test and sampling validation patches "
                f"from train city {train_val_city}."
            )
            warn(warning)
            selected["warnings"].append(warning)

    return split_df, selected


# ---------------------------------------------------------------------
# Validation and summaries
# ---------------------------------------------------------------------

def validate_split(df: pd.DataFrame) -> Dict[str, Any]:
    split_names = ["train", "val", "test"]

    problems: List[str] = []

    split_counts = df["split"].value_counts().to_dict()

    for split in split_names:
        if split_counts.get(split, 0) == 0:
            problems.append(f"Split {split} is empty.")

    split_region_sets = {
        split: set(df[df["split"] == split]["region"].unique().tolist())
        for split in split_names
    }

    for split in split_names:
        missing = sorted(set(REGION_ORDER) - split_region_sets[split])
        if missing:
            problems.append(f"Split {split} is missing regions: {missing}")

    split_patch_sets = {
        split: set(df[df["split"] == split]["patch_id"].astype(str).unique().tolist())
        for split in split_names
    }

    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            overlap = split_patch_sets[a].intersection(split_patch_sets[b])
            if overlap:
                problems.append(
                    f"Patch leakage between {a} and {b}: {len(overlap)}. "
                    f"First examples: {sorted(list(overlap))[:10]}"
                )

    # Test cities must be fully held out from train and val.
    test_cities = set(df[df["split"] == "test"]["city"].unique().tolist())
    train_cities = set(df[df["split"] == "train"]["city"].unique().tolist())
    val_cities = set(df[df["split"] == "val"]["city"].unique().tolist())

    test_train_overlap = sorted(test_cities.intersection(train_cities))
    test_val_overlap = sorted(test_cities.intersection(val_cities))

    if test_train_overlap:
        problems.append(f"Test cities also appear in train: {test_train_overlap}")

    if test_val_overlap:
        problems.append(f"Test cities also appear in val: {test_val_overlap}")

    # Train/val city overlap is allowed only for patch-level validation source cities.
    train_val_overlap = sorted(train_cities.intersection(val_cities))

    allowed_train_val_overlap = sorted(
        df[df["split_assignment_type"] == "val_patch_sample_from_train_city"]["city"]
        .unique()
        .tolist()
    )

    unexpected_train_val_overlap = sorted(set(train_val_overlap) - set(allowed_train_val_overlap))

    if unexpected_train_val_overlap:
        problems.append(
            "Unexpected train/val city overlap outside patch-level validation source cities: "
            f"{unexpected_train_val_overlap}"
        )

    if problems:
        fail("Split validation failed:\n" + "\n".join(f"- {p}" for p in problems))

    return {
        "status": "ok",
        "split_patch_counts": {k: int(v) for k, v in split_counts.items()},
        "split_region_counts": {
            split: int(len(split_region_sets[split]))
            for split in split_names
        },
        "split_regions": {
            split: sorted(split_region_sets[split])
            for split in split_names
        },
        "test_city_overlap_with_train_or_val": "none",
        "train_val_city_overlap": train_val_overlap,
        "allowed_train_val_city_overlap": allowed_train_val_overlap,
        "patch_overlap": "none",
        "interpretation": (
            "Train/val city overlap is allowed only for regions with two cities, "
            "where validation patches are sampled from the remaining train city to "
            "preserve train-region coverage."
        ),
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
    summary["positive_patch_pct"] = 100.0 * summary["n_positive_patches"] / summary["n_patches"].clip(lower=1)
    summary["total_pixels"] = summary["n_patches"] * patch_area
    summary["label_positive_percent"] = (
        100.0 * summary["label_positive_pixels"] / summary["total_pixels"].clip(lower=1)
    )

    split_order = {"train": 0, "val": 1, "test": 2}
    summary["_order"] = summary["split"].map(split_order).fillna(99).astype(int)

    return summary.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def build_split_city_summary(df: pd.DataFrame, patch_size: int) -> pd.DataFrame:
    patch_area = int(patch_size) * int(patch_size)

    summary = (
        df.groupby(["split", "region", "city", "split_assignment_type"])
        .agg(
            n_patches=("patch_id", "count"),
            n_positive_patches=("label_binary", "sum"),
            label_positive_pixels=("label_positive_pixels", "sum"),
        )
        .reset_index()
    )

    summary["positive_patch_pct"] = 100.0 * summary["n_positive_patches"] / summary["n_patches"].clip(lower=1)
    summary["total_pixels"] = summary["n_patches"] * patch_area
    summary["label_positive_percent"] = (
        100.0 * summary["label_positive_pixels"] / summary["total_pixels"].clip(lower=1)
    )

    split_order = {"train": 0, "val": 1, "test": 2}
    region_order = {region: i for i, region in enumerate(REGION_ORDER)}

    summary["_split_order"] = summary["split"].map(split_order).fillna(99).astype(int)
    summary["_region_order"] = summary["region"].map(region_order).fillna(999).astype(int)

    return (
        summary.sort_values(["_split_order", "_region_order", "region", "city"])
        .drop(columns=["_split_order", "_region_order"])
        .reset_index(drop=True)
    )


def build_selected_city_summary(city_stats: pd.DataFrame, selected: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for region, info in selected["regions"].items():
        test_city = info["test_city"]

        rows.append(
            {
                "selected_role": "test_city_holdout",
                "region": region,
                "city": test_city,
                "mode": info["mode"],
            }
        )

        if "val_city" in info:
            rows.append(
                {
                    "selected_role": "val_city_holdout",
                    "region": region,
                    "city": info["val_city"],
                    "mode": info["mode"],
                }
            )

        if "patch_val_source_city" in info:
            rows.append(
                {
                    "selected_role": "patch_val_source_train_city",
                    "region": region,
                    "city": info["patch_val_source_city"],
                    "mode": info["mode"],
                }
            )

    selected_df = pd.DataFrame(rows)

    out = selected_df.merge(
        city_stats,
        on=["region", "city"],
        how="left",
    )

    region_order = {region: i for i, region in enumerate(REGION_ORDER)}
    role_order = {
        "test_city_holdout": 0,
        "val_city_holdout": 1,
        "patch_val_source_train_city": 2,
    }

    out["_region_order"] = out["region"].map(region_order).fillna(999).astype(int)
    out["_role_order"] = out["selected_role"].map(role_order).fillna(999).astype(int)

    return (
        out.sort_values(["_region_order", "_role_order", "city"])
        .drop(columns=["_region_order", "_role_order"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def write_split_report(
    path: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    output_dir: Path,
    selected_city_summary: pd.DataFrame,
    split_patch_summary: pd.DataFrame,
    split_city_summary: pd.DataFrame,
    validation: Dict[str, Any],
    selected: Dict[str, Any],
) -> None:
    lines: List[str] = []

    lines.append("# Region-Balanced Train-Covered City Split Report")
    lines.append("")
    lines.append(f"Created UTC: `{now_utc()}`")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Create a simpler split for BigEarthNet/reBEN segmentation experiments. "
        "The main constraint is that the training split must contain examples from all five Brazilian regions."
    )
    lines.append("")
    lines.append("## Split Design")
    lines.append("")
    lines.append("For regions with at least three cities:")
    lines.append("")
    lines.append("- one city is held out for test;")
    lines.append("- one different city is held out for validation;")
    lines.append("- all remaining cities stay in training.")
    lines.append("")
    lines.append("For regions with exactly two cities:")
    lines.append("")
    lines.append("- one city is held out for test;")
    lines.append("- the other city remains in training;")
    lines.append("- validation patches are sampled from that train city.")
    lines.append("")
    lines.append(
        "This means validation is not purely city-held-out for two-city regions, "
        "but this is intentional because the current goal is learnability testing, not strict geographic generalization."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Instance root: `{args.instance_root}`")
    lines.append(f"- Manifest path: `{path_to_str(manifest_path)}`")
    lines.append(f"- Output directory: `{path_to_str(output_dir)}`")
    lines.append(f"- Target modality: `{args.target_modality}`")
    lines.append(f"- Patch size: `{args.patch_size}`")
    lines.append(f"- Split design: `{args.split_design}`")
    lines.append(f"- City ranking: `{args.city_ranking}`")
    lines.append(f"- Validation patch fraction for two-city regions: `{args.val_patch_fraction}`")
    lines.append(f"- Minimum patch-level validation patches: `{args.min_patch_val_patches}`")
    lines.append(f"- Maximum patch-level validation patches: `{args.max_patch_val_patches}`")
    lines.append("")
    lines.append("## Selected Cities")
    lines.append("")
    lines.append(safe_markdown_table(selected_city_summary))
    lines.append("")
    lines.append("## Split-Level Summary")
    lines.append("")
    lines.append(safe_markdown_table(split_patch_summary))
    lines.append("")
    lines.append("## Split-City Summary")
    lines.append("")
    lines.append(safe_markdown_table(split_city_summary))
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
    lines.append(json.dumps(jsonable(selected), indent=2, ensure_ascii=False)[:20000])
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This split should be used for the next learnability experiment. "
        "If the BigEarthNet/reBEN model performs reasonably here but poorly under LORO, "
        "then the dataset contains learnable signal and the LORO difficulty is likely caused by geographic domain shift. "
        "If performance remains poor even here, then the issue may be related to labels, alignment, normalization, "
        "class imbalance, model architecture, or input modality handling."
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

    banner("Build train-region-covered city split for BigEarthNet/reBEN experiments")

    log("INFO", f"Instance root:      {path_to_str(instance_root)}")
    log("INFO", f"Manifest path:      {path_to_str(manifest_path)}")
    log("INFO", f"Output directory:   {path_to_str(output_dir)}")
    log("INFO", f"Target modality:    {args.target_modality}")
    log("INFO", f"Split design:       {args.split_design}")
    log("INFO", f"City ranking:       {args.city_ranking}")

    if args.split_design != "train_region_covered":
        fail("This updated script currently supports only --split-design train_region_covered.")

    ensure_output_dir(output_dir, overwrite=bool(args.overwrite))

    df = load_manifest(
        manifest_path=manifest_path,
        target_modality=str(args.target_modality),
        allow_duplicate_patch_ids=bool(args.allow_duplicate_patch_ids),
    )

    city_stats = build_city_level_stats(
        df=df,
        patch_size=int(args.patch_size),
    )

    print_city_stats_summary(city_stats)

    split_df, selected = build_train_region_covered_split(
        df=df,
        city_stats=city_stats,
        city_ranking=str(args.city_ranking),
        val_patch_fraction=float(args.val_patch_fraction),
        min_patch_val_patches=int(args.min_patch_val_patches),
        max_patch_val_patches=int(args.max_patch_val_patches),
        seed=int(args.seed),
    )

    validation = validate_split(split_df)

    selected_city_summary = build_selected_city_summary(
        city_stats=city_stats,
        selected=selected,
    )

    split_patch_summary = build_split_patch_summary(
        df=split_df,
        patch_size=int(args.patch_size),
    )

    split_city_summary = build_split_city_summary(
        df=split_df,
        patch_size=int(args.patch_size),
    )

    city_stats_path = output_dir / "city_level_stats.csv"
    selected_json_path = output_dir / "selected_cities.json"
    selected_city_summary_path = output_dir / "selected_city_summary.csv"
    split_patch_summary_path = output_dir / "split_patch_summary.csv"
    split_city_summary_path = output_dir / "split_city_summary.csv"
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    report_path = output_dir / "split_report.md"

    city_stats.to_csv(city_stats_path, index=False)
    selected_city_summary.to_csv(selected_city_summary_path, index=False)
    split_patch_summary.to_csv(split_patch_summary_path, index=False)
    split_city_summary.to_csv(split_city_summary_path, index=False)

    split_df[split_df["split"] == "train"].to_csv(train_path, index=False)
    split_df[split_df["split"] == "val"].to_csv(val_path, index=False)
    split_df[split_df["split"] == "test"].to_csv(test_path, index=False)

    payload = {
        "created_utc": now_utc(),
        "instance_root": path_to_str(instance_root),
        "manifest_path": path_to_str(manifest_path),
        "output_dir": path_to_str(output_dir),
        "target_modality": str(args.target_modality),
        "patch_size": int(args.patch_size),
        "split_design": str(args.split_design),
        "city_ranking": str(args.city_ranking),
        "selection": selected,
        "validation": validation,
        "outputs": {
            "city_level_stats_csv": path_to_str(city_stats_path),
            "selected_city_summary_csv": path_to_str(selected_city_summary_path),
            "split_patch_summary_csv": path_to_str(split_patch_summary_path),
            "split_city_summary_csv": path_to_str(split_city_summary_path),
            "train_csv": path_to_str(train_path),
            "val_csv": path_to_str(val_path),
            "test_csv": path_to_str(test_path),
            "report_md": path_to_str(report_path),
        },
    }

    write_json(selected_json_path, payload)

    write_split_report(
        path=report_path,
        args=args,
        manifest_path=manifest_path,
        output_dir=output_dir,
        selected_city_summary=selected_city_summary,
        split_patch_summary=split_patch_summary,
        split_city_summary=split_city_summary,
        validation=validation,
        selected=selected,
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

    log("INFO", "Selected cities / roles:")
    print(
        selected_city_summary[
            [
                "selected_role",
                "region",
                "city",
                "mode",
                "n_patches",
                "n_positive_patches",
                "positive_patch_pct",
                "label_positive_percent",
            ]
        ].to_string(index=False),
        flush=True,
    )

    log("INFO", "Validation:")
    print(json.dumps(jsonable(validation), indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train-region-covered splits for BigEarthNet/reBEN favela segmentation experiments."
    )

    parser.add_argument(
        "--instance-root",
        required=True,
        help="Dataset instance root.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional explicit manifest path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--target-modality",
        default="s2_s1_snap_vv_vh",
        help="Manifest modality to use.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size.",
    )

    parser.add_argument(
        "--split-design",
        choices=["train_region_covered"],
        default="train_region_covered",
        help="Split design. Current supported value: train_region_covered.",
    )
    parser.add_argument(
        "--city-ranking",
        choices=["highest_positive", "balanced_positive_percent", "largest_patch_count"],
        default="highest_positive",
        help="How to rank candidate cities inside each region.",
    )

    parser.add_argument(
        "--val-patch-fraction",
        type=float,
        default=0.20,
        help="Fraction of patches sampled for validation from train city in two-city regions.",
    )
    parser.add_argument(
        "--min-patch-val-patches",
        type=int,
        default=50,
        help="Minimum number of patch-level validation patches for two-city regions.",
    )
    parser.add_argument(
        "--max-patch-val-patches",
        type=int,
        default=250,
        help="Maximum number of patch-level validation patches for two-city regions.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patch-level validation sampling.",
    )
    parser.add_argument(
        "--allow-duplicate-patch-ids",
        action="store_true",
        help="If duplicate patch_id values exist after modality filtering, keep first occurrence.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())