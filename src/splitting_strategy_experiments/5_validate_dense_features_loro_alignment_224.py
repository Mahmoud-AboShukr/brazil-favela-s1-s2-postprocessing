#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
5_validate_dense_features_loro_alignment_224.py

Main objective
--------------
Validate alignment between:

1. Full dense CROMA feature file:
       N x 768 x 28 x 28

2. Dense CROMA metadata / manifest:
       patch_id -> dense_index

3. Leave-One-Region-Out split manifests:
       train / val / test CSV files per held-out region

Why this script exists
----------------------
Before training UPerNet, we must prove that every split row can be mapped to
the correct dense CROMA feature index.

This script does NOT train a model.
It prepares trustworthy, training-ready split CSVs with a dense_index column.

Recommended command
-------------------
python src/splitting_strategy_experiments/5_validate_dense_features_loro_alignment_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --target-modality "s2_s1_snap_vv_vh" `
  --feature-key "joint_encodings" `
  --output-dtype "float16" `
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EXPECTED_FEATURE_SHAPE = (12699, 768, 28, 28)

EXPECTED_REGIONS = [
    "Central-West",
    "North",
    "Northeast",
    "South",
    "Southeast",
]

SPLIT_NAMES = ["train", "val", "test"]


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    log("ERROR", message)
    raise SystemExit(exit_code)


def warn(message: str) -> None:
    log("WARNING", message)


def path_to_str(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return str(path).replace("\\", "/")


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted([jsonable(v) for v in value])
    if isinstance(value, Path):
        return path_to_str(value)
    if isinstance(value, np.ndarray):
        if value.size <= 20:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "preview": value.reshape(-1)[:20].tolist(),
        }
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if math.isnan(v):
            return None
        return v
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    return value


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)

    existing = (
        list(path.glob("*.csv"))
        + list(path.glob("*.json"))
        + list(path.glob("*.md"))
    )

    if existing and not overwrite:
        fail(
            f"Output directory already contains files:\n{path_to_str(path)}\n\n"
            f"Use --overwrite to replace validation outputs."
        )


def write_json(path: Path, payload: Dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(f"JSON output exists and --overwrite was not used:\n{path_to_str(path)}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------

def dense_feature_stem(
    target_modality: str,
    feature_key: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
    output_dtype: str,
) -> str:
    return (
        f"croma_dense_features_{target_modality}_{feature_key}_"
        f"ps{patch_size}_st{stride}_{edge_mode}_{output_dtype}"
    )


def default_dense_dir(instance_root: Path, patch_size: int, stride: int, edge_mode: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "splitting_strategy_experiments"
        / f"croma_dense_features_ps{patch_size}_st{stride}_{edge_mode}"
    )


def default_dense_feature_path(
    instance_root: Path,
    target_modality: str,
    feature_key: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
    output_dtype: str,
) -> Path:
    stem = dense_feature_stem(
        target_modality=target_modality,
        feature_key=feature_key,
        patch_size=patch_size,
        stride=stride,
        edge_mode=edge_mode,
        output_dtype=output_dtype,
    )
    return default_dense_dir(instance_root, patch_size, stride, edge_mode) / f"{stem}.npy"


def default_dense_metadata_path(
    instance_root: Path,
    target_modality: str,
    feature_key: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
    output_dtype: str,
) -> Path:
    stem = dense_feature_stem(
        target_modality=target_modality,
        feature_key=feature_key,
        patch_size=patch_size,
        stride=stride,
        edge_mode=edge_mode,
        output_dtype=output_dtype,
    )
    return default_dense_dir(instance_root, patch_size, stride, edge_mode) / f"{stem}_metadata.npz"


def default_dense_manifest_path(
    instance_root: Path,
    target_modality: str,
    feature_key: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
    output_dtype: str,
) -> Path:
    stem = dense_feature_stem(
        target_modality=target_modality,
        feature_key=feature_key,
        patch_size=patch_size,
        stride=stride,
        edge_mode=edge_mode,
        output_dtype=output_dtype,
    )
    return default_dense_dir(instance_root, patch_size, stride, edge_mode) / f"{stem}_manifest.csv"


def default_loro_split_dir(instance_root: Path, patch_size: int, stride: int, edge_mode: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "upernet_croma"
        / f"splits_loro_ps{patch_size}_st{stride}_{edge_mode}"
    )


def default_output_dir(instance_root: Path, patch_size: int, stride: int, edge_mode: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "splitting_strategy_experiments"
        / f"dense_loro_alignment_validation_ps{patch_size}_st{stride}_{edge_mode}"
    )


def slug_region(region: str) -> str:
    text = str(region).strip()
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text


# ---------------------------------------------------------------------
# Dense feature validation
# ---------------------------------------------------------------------

def inspect_dense_feature_file(path: Path) -> Dict[str, Any]:
    log("STEP", f"Inspecting dense feature file:\n{path_to_str(path)}")

    if not path.exists():
        fail(f"Dense feature file does not exist:\n{path_to_str(path)}")

    x = np.load(path, mmap_mode="r")

    info = {
        "path": path_to_str(path),
        "shape": tuple(int(v) for v in x.shape),
        "dtype": str(x.dtype),
        "ndim": int(x.ndim),
        "n_patches": int(x.shape[0]) if x.ndim >= 1 else 0,
        "channels": int(x.shape[1]) if x.ndim >= 2 else None,
        "height": int(x.shape[2]) if x.ndim >= 3 else None,
        "width": int(x.shape[3]) if x.ndim >= 4 else None,
    }

    log("INFO", f"Dense feature shape: {info['shape']}")
    log("INFO", f"Dense feature dtype:  {info['dtype']}")

    if info["ndim"] != 4:
        fail(f"Dense feature file must be 4D N x C x H x W, got shape {info['shape']}")

    if info["channels"] != 768:
        warn(f"Expected 768 channels, got {info['channels']}.")

    if info["height"] != 28 or info["width"] != 28:
        warn(f"Expected 28x28 dense grid, got {info['height']}x{info['width']}.")

    return info


def read_metadata_npz(path: Path) -> Dict[str, Any]:
    log("STEP", f"Reading dense metadata NPZ:\n{path_to_str(path)}")

    if not path.exists():
        fail(f"Dense metadata NPZ does not exist:\n{path_to_str(path)}")

    metadata: Dict[str, Any] = {
        "path": path_to_str(path),
        "keys": [],
        "arrays": {},
    }

    with np.load(path, allow_pickle=False) as data:
        metadata["keys"] = list(data.keys())

        for key in data.keys():
            arr = data[key]
            metadata["arrays"][key] = {
                "shape": tuple(int(v) for v in arr.shape),
                "dtype": str(arr.dtype),
            }

        required = ["patch_ids", "cities", "regions", "label_binary"]
        missing = [k for k in required if k not in data.keys()]
        if missing:
            fail(f"Metadata NPZ is missing required keys: {missing}")

        patch_ids = data["patch_ids"].astype(str)
        cities = data["cities"].astype(str)
        regions = data["regions"].astype(str)
        label_binary = data["label_binary"].astype(np.int64)

        metadata["patch_ids"] = patch_ids
        metadata["cities"] = cities
        metadata["regions"] = regions
        metadata["label_binary"] = label_binary

        optional_keys = [
            "manifest_row_ids",
            "label_positive_pixels",
            "label_positive_percent",
            "label_density_bins",
            "dense_feature_shape",
            "feature_key",
            "target_modality",
        ]

        for key in optional_keys:
            if key in data.keys():
                metadata[key] = data[key].copy()

    log("INFO", f"Metadata patch count: {len(metadata['patch_ids']):,}")
    log("INFO", f"Metadata keys: {metadata['keys']}")

    return metadata


def read_dense_manifest(path: Path) -> pd.DataFrame:
    log("STEP", f"Reading dense manifest CSV:\n{path_to_str(path)}")

    if not path.exists():
        fail(f"Dense manifest CSV does not exist:\n{path_to_str(path)}")

    df = pd.read_csv(path)

    if df.empty:
        fail("Dense manifest CSV is empty.")

    required = ["dense_index", "patch_id", "city", "region"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(
            f"Dense manifest CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["dense_index"] = pd.to_numeric(df["dense_index"], errors="coerce").astype("Int64")
    df["patch_id"] = df["patch_id"].astype(str)

    log("INFO", f"Dense manifest rows: {len(df):,}")

    return df


def validate_dense_consistency(
    feature_info: Dict[str, Any],
    metadata: Dict[str, Any],
    dense_manifest: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    n_features = int(feature_info["n_patches"])
    n_meta = int(len(metadata["patch_ids"]))
    n_manifest = int(len(dense_manifest))

    if n_features != n_meta:
        errors.append(f"Dense feature N={n_features:,} does not match metadata patch count={n_meta:,}.")

    if n_features != n_manifest:
        errors.append(f"Dense feature N={n_features:,} does not match dense manifest rows={n_manifest:,}.")

    expected_indices = np.arange(n_manifest, dtype=np.int64)
    actual_indices = dense_manifest["dense_index"].astype(np.int64).values

    if not np.array_equal(actual_indices, expected_indices):
        errors.append("Dense manifest dense_index is not exactly 0..N-1 in row order.")

    manifest_patch_ids = dense_manifest["patch_id"].astype(str).values
    metadata_patch_ids = metadata["patch_ids"].astype(str)

    if len(manifest_patch_ids) == len(metadata_patch_ids):
        if not np.array_equal(manifest_patch_ids, metadata_patch_ids):
            errors.append("Patch ID order mismatch between dense manifest and metadata NPZ.")
    else:
        errors.append("Cannot compare metadata and manifest patch IDs because lengths differ.")

    duplicated_manifest_patch_ids = dense_manifest["patch_id"].duplicated().sum()
    if duplicated_manifest_patch_ids > 0:
        errors.append(f"Dense manifest contains {duplicated_manifest_patch_ids:,} duplicated patch IDs.")

    duplicated_metadata_patch_ids = pd.Series(metadata_patch_ids).duplicated().sum()
    if duplicated_metadata_patch_ids > 0:
        errors.append(f"Metadata NPZ contains {duplicated_metadata_patch_ids:,} duplicated patch IDs.")

    label_binary = metadata["label_binary"]
    if len(label_binary) == n_features:
        pos = int(np.asarray(label_binary).sum())
        if pos <= 0:
            warnings.append("Metadata label_binary contains no positive patches.")
    else:
        errors.append("metadata label_binary length does not match dense feature N.")

    mapping_df = dense_manifest[["dense_index", "patch_id", "city", "region"]].copy()

    optional_cols = [
        "label_binary",
        "label_positive_pixels",
        "label_positive_percent",
        "label_density_bin",
    ]

    for col in optional_cols:
        if col in dense_manifest.columns:
            mapping_df[col] = dense_manifest[col].values

    return mapping_df, errors, warnings


# ---------------------------------------------------------------------
# LORO split validation
# ---------------------------------------------------------------------

def discover_loro_split_files(split_dir: Path) -> Dict[str, Dict[str, Path]]:
    log("STEP", f"Discovering LORO split files:\n{path_to_str(split_dir)}")

    if not split_dir.exists():
        fail(f"LORO split directory does not exist:\n{path_to_str(split_dir)}")

    files = sorted(split_dir.glob("loro_fold_*_*.csv"))

    if not files:
        fail(f"No LORO split CSV files found in:\n{path_to_str(split_dir)}")

    pattern = re.compile(r"^loro_fold_(.+)_(train|val|test)\.csv$")

    folds: Dict[str, Dict[str, Path]] = {}

    for path in files:
        m = pattern.match(path.name)
        if not m:
            continue

        region_slug = m.group(1)
        split_name = m.group(2)

        folds.setdefault(region_slug, {})[split_name] = path

    if not folds:
        fail("No valid LORO fold split files were discovered.")

    log("INFO", f"Discovered fold groups: {sorted(folds.keys())}")

    for region_slug, split_paths in sorted(folds.items()):
        missing = [s for s in SPLIT_NAMES if s not in split_paths]
        if missing:
            fail(f"Fold {region_slug} is missing split files: {missing}")

    return folds


def infer_heldout_region_from_slug(region_slug: str) -> str:
    # Existing filenames use Central-West, North, Northeast, South, Southeast.
    for region in EXPECTED_REGIONS:
        if slug_region(region) == region_slug:
            return region

    # Fallback: convert underscores to spaces.
    return region_slug.replace("_", " ")


def read_split(path: Path, split_name: str, heldout_region: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.empty:
        fail(f"LORO split is empty:\n{path_to_str(path)}")

    if "patch_id" not in df.columns:
        fail(
            f"LORO split missing patch_id column:\n{path_to_str(path)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["patch_id"] = df["patch_id"].astype(str)
    df["expected_split"] = split_name
    df["expected_heldout_region"] = heldout_region

    return df


def validate_single_fold(
    heldout_region: str,
    split_paths: Dict[str, Path],
    mapping_df: pd.DataFrame,
    output_dir: Path,
    overwrite: bool,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    summary_rows: List[Dict[str, Any]] = []

    mapping_small = mapping_df[["patch_id", "dense_index"]].copy()

    split_dfs: Dict[str, pd.DataFrame] = {}
    enriched_dfs: Dict[str, pd.DataFrame] = {}

    for split_name in SPLIT_NAMES:
        split_path = split_paths[split_name]
        df = read_split(split_path, split_name, heldout_region)
        split_dfs[split_name] = df

        merged = df.merge(mapping_small, on="patch_id", how="left", validate="many_to_one")
        missing_dense = int(merged["dense_index"].isna().sum())

        if missing_dense > 0:
            errors.append(
                f"{heldout_region}/{split_name}: {missing_dense:,} rows do not have a dense_index."
            )

        if merged["patch_id"].duplicated().sum() > 0:
            warnings.append(
                f"{heldout_region}/{split_name}: split contains duplicated patch IDs."
            )

        if "upernet_split" in merged.columns:
            bad_split_values = merged["upernet_split"].astype(str).ne(split_name).sum()
            if bad_split_values > 0:
                errors.append(
                    f"{heldout_region}/{split_name}: {bad_split_values:,} rows have wrong upernet_split."
                )

        if "loro_heldout_region" in merged.columns:
            bad_heldout_values = merged["loro_heldout_region"].astype(str).ne(heldout_region).sum()
            if bad_heldout_values > 0:
                errors.append(
                    f"{heldout_region}/{split_name}: {bad_heldout_values:,} rows have wrong loro_heldout_region."
                )

        if "region" in merged.columns:
            regions = set(merged["region"].astype(str).unique())

            if split_name == "test":
                if regions != {heldout_region}:
                    errors.append(
                        f"{heldout_region}/test: expected only region {heldout_region}, found {sorted(regions)}."
                    )
            else:
                if heldout_region in regions:
                    errors.append(
                        f"{heldout_region}/{split_name}: held-out region appears outside test split."
                    )

        label_positive_pixels = None
        if "label_positive_pixels" in merged.columns:
            label_positive_pixels = pd.to_numeric(
                merged["label_positive_pixels"], errors="coerce"
            ).fillna(0)

        label_binary = None
        if "label_binary" in merged.columns:
            label_binary = pd.to_numeric(
                merged["label_binary"], errors="coerce"
            ).fillna(0)

        if label_positive_pixels is not None:
            positive_patches = int((label_positive_pixels > 0).sum())
        elif label_binary is not None:
            positive_patches = int((label_binary > 0).sum())
        else:
            positive_patches = -1

        n_rows = int(len(merged))
        n_unique_patch_ids = int(merged["patch_id"].nunique())
        n_unique_dense_indices = int(merged["dense_index"].dropna().nunique())

        city_count = int(merged["city"].nunique()) if "city" in merged.columns else -1
        region_values = (
            sorted(merged["region"].astype(str).unique().tolist())
            if "region" in merged.columns
            else []
        )

        summary_rows.append(
            {
                "heldout_region": heldout_region,
                "split": split_name,
                "source_csv": path_to_str(split_path),
                "rows": n_rows,
                "unique_patch_ids": n_unique_patch_ids,
                "unique_dense_indices": n_unique_dense_indices,
                "missing_dense_index": missing_dense,
                "positive_patches": positive_patches,
                "empty_patches": n_rows - positive_patches if positive_patches >= 0 else -1,
                "positive_patch_percent": (
                    float(100.0 * positive_patches / n_rows)
                    if positive_patches >= 0 and n_rows > 0
                    else None
                ),
                "n_cities": city_count,
                "regions": ";".join(region_values),
            }
        )

        # Save enriched split for direct use by future PyTorch dataset.
        region_slug = slug_region(heldout_region)
        out_path = output_dir / f"loro_fold_{region_slug}_{split_name}_with_dense_index.csv"

        if out_path.exists() and not overwrite:
            fail(f"Output exists and --overwrite was not used:\n{path_to_str(out_path)}")

        merged.to_csv(out_path, index=False)
        enriched_dfs[split_name] = merged

    # Patch/dense-index leakage checks.
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        a_patch = set(enriched_dfs[a]["patch_id"].astype(str))
        b_patch = set(enriched_dfs[b]["patch_id"].astype(str))
        overlap_patch = a_patch & b_patch

        if overlap_patch:
            errors.append(
                f"{heldout_region}: patch_id leakage between {a} and {b}: {len(overlap_patch):,} overlapping patches."
            )

        a_dense = set(enriched_dfs[a]["dense_index"].dropna().astype(int))
        b_dense = set(enriched_dfs[b]["dense_index"].dropna().astype(int))
        overlap_dense = a_dense & b_dense

        if overlap_dense:
            errors.append(
                f"{heldout_region}: dense_index leakage between {a} and {b}: {len(overlap_dense):,} overlapping indices."
            )

    # City-disjoint validation checks.
    if "city" in enriched_dfs["train"].columns and "city" in enriched_dfs["val"].columns:
        train_cities = set(enriched_dfs["train"]["city"].astype(str))
        val_cities = set(enriched_dfs["val"]["city"].astype(str))
        test_cities = set(enriched_dfs["test"]["city"].astype(str))

        train_val_city_overlap = train_cities & val_cities
        if train_val_city_overlap:
            errors.append(
                f"{heldout_region}: city leakage between train and val: {sorted(train_val_city_overlap)}"
            )

        train_test_city_overlap = train_cities & test_cities
        val_test_city_overlap = val_cities & test_cities

        if train_test_city_overlap:
            errors.append(
                f"{heldout_region}: city overlap between train and test: {sorted(train_test_city_overlap)}"
            )

        if val_test_city_overlap:
            errors.append(
                f"{heldout_region}: city overlap between val and test: {sorted(val_test_city_overlap)}"
            )

    return summary_rows, errors, warnings


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def write_markdown_report(
    path: Path,
    payload: Dict[str, Any],
    split_summary_df: pd.DataFrame,
    errors: List[str],
    warnings: List[str],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        fail(f"Markdown report exists and --overwrite was not used:\n{path_to_str(path)}")

    lines: List[str] = []

    lines.append("# Dense CROMA Features and LORO Split Alignment Validation")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Validate that the full dense CROMA feature tensor, dense feature metadata, "
        "dense manifest, and Leave-One-Region-Out split manifests are perfectly aligned."
    )
    lines.append("")
    lines.append("## Final status")
    lines.append("")
    lines.append(f"- Status: `{payload['status']}`")
    lines.append(f"- Error count: `{len(errors)}`")
    lines.append(f"- Warning count: `{len(warnings)}`")
    lines.append("")
    lines.append("## Dense feature file")
    lines.append("")
    feature_info = payload["dense_feature_info"]
    lines.append(f"- Path: `{feature_info['path']}`")
    lines.append(f"- Shape: `{feature_info['shape']}`")
    lines.append(f"- Dtype: `{feature_info['dtype']}`")
    lines.append("")
    lines.append("## Dense metadata and manifest")
    lines.append("")
    lines.append(f"- Metadata NPZ: `{payload['dense_metadata_path']}`")
    lines.append(f"- Dense manifest CSV: `{payload['dense_manifest_path']}`")
    lines.append(f"- Mapping CSV: `{payload['mapping_csv_path']}`")
    lines.append("")
    lines.append("## Split summary")
    lines.append("")
    lines.append(
        "| Held-out region | Split | Rows | Unique dense indices | Missing dense index | Positive patches | Positive % | Cities | Regions |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---|"
    )

    for _, row in split_summary_df.iterrows():
        pos_pct = row["positive_patch_percent"]
        pos_pct_str = "NA" if pd.isna(pos_pct) else f"{float(pos_pct):.3f}"

        lines.append(
            f"| {row['heldout_region']} "
            f"| {row['split']} "
            f"| {int(row['rows']):,} "
            f"| {int(row['unique_dense_indices']):,} "
            f"| {int(row['missing_dense_index']):,} "
            f"| {int(row['positive_patches']):,} "
            f"| {pos_pct_str} "
            f"| {int(row['n_cities']) if int(row['n_cities']) >= 0 else 'NA'} "
            f"| {row['regions']} |"
        )

    lines.append("")
    lines.append("## Errors")
    lines.append("")
    if errors:
        for error in errors:
            lines.append(f"- {error}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if payload["status"] == "passed":
        lines.append(
            "The dense CROMA features are aligned with all Leave-One-Region-Out split files. "
            "The enriched split CSVs with `dense_index` can now be used by the UPerNet training dataset."
        )
    else:
        lines.append(
            "The alignment validation failed. Do not train UPerNet until the errors above are resolved."
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_validation(args: argparse.Namespace) -> None:
    instance_root = Path(args.instance_root)

    dense_feature_path = (
        Path(args.dense_feature_path)
        if args.dense_feature_path
        else default_dense_feature_path(
            instance_root=instance_root,
            target_modality=args.target_modality,
            feature_key=args.feature_key,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            output_dtype=args.output_dtype,
        )
    )

    dense_metadata_path = (
        Path(args.dense_metadata_path)
        if args.dense_metadata_path
        else default_dense_metadata_path(
            instance_root=instance_root,
            target_modality=args.target_modality,
            feature_key=args.feature_key,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            output_dtype=args.output_dtype,
        )
    )

    dense_manifest_path = (
        Path(args.dense_manifest_path)
        if args.dense_manifest_path
        else default_dense_manifest_path(
            instance_root=instance_root,
            target_modality=args.target_modality,
            feature_key=args.feature_key,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            output_dtype=args.output_dtype,
        )
    )

    loro_split_dir = (
        Path(args.loro_split_dir)
        if args.loro_split_dir
        else default_loro_split_dir(
            instance_root=instance_root,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
        )
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(
            instance_root=instance_root,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
        )
    )

    log("=" * 100, "")
    log("STEP", "Dense feature and LORO alignment validation")
    log("INFO", f"Instance root:       {path_to_str(instance_root)}")
    log("INFO", f"Dense feature path:  {path_to_str(dense_feature_path)}")
    log("INFO", f"Dense metadata path: {path_to_str(dense_metadata_path)}")
    log("INFO", f"Dense manifest path: {path_to_str(dense_manifest_path)}")
    log("INFO", f"LORO split dir:      {path_to_str(loro_split_dir)}")
    log("INFO", f"Output dir:          {path_to_str(output_dir)}")
    log("=" * 100, "")

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    all_errors: List[str] = []
    all_warnings: List[str] = []

    feature_info = inspect_dense_feature_file(dense_feature_path)
    metadata = read_metadata_npz(dense_metadata_path)
    dense_manifest = read_dense_manifest(dense_manifest_path)

    mapping_df, dense_errors, dense_warnings = validate_dense_consistency(
        feature_info=feature_info,
        metadata=metadata,
        dense_manifest=dense_manifest,
    )

    all_errors.extend(dense_errors)
    all_warnings.extend(dense_warnings)

    mapping_csv_path = output_dir / f"dense_patch_id_to_index_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    mapping_df.to_csv(mapping_csv_path, index=False)
    log("OK", f"Wrote patch_id to dense_index mapping:\n{path_to_str(mapping_csv_path)}")

    folds = discover_loro_split_files(loro_split_dir)

    split_summary_rows: List[Dict[str, Any]] = []

    for region_slug, split_paths in sorted(folds.items()):
        heldout_region = infer_heldout_region_from_slug(region_slug)

        log("STEP", f"Validating LORO fold: {heldout_region}")

        fold_rows, fold_errors, fold_warnings = validate_single_fold(
            heldout_region=heldout_region,
            split_paths=split_paths,
            mapping_df=mapping_df,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )

        split_summary_rows.extend(fold_rows)
        all_errors.extend(fold_errors)
        all_warnings.extend(fold_warnings)

        if fold_errors:
            log("ERROR", f"{heldout_region}: {len(fold_errors)} errors")
        else:
            log("OK", f"{heldout_region}: alignment checks passed")

    split_summary_df = pd.DataFrame(split_summary_rows)

    split_summary_csv_path = output_dir / f"dense_loro_alignment_split_summary_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    split_summary_df.to_csv(split_summary_csv_path, index=False)

    status = "passed" if not all_errors else "failed"

    summary_json_path = output_dir / f"dense_loro_alignment_validation_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.json"
    report_md_path = output_dir / f"dense_loro_alignment_validation_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.md"

    payload = {
        "status": status,
        "instance_root": path_to_str(instance_root),
        "target_modality": args.target_modality,
        "feature_key": args.feature_key,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "edge_mode": args.edge_mode,
        "output_dtype": args.output_dtype,
        "dense_feature_info": feature_info,
        "dense_metadata_path": path_to_str(dense_metadata_path),
        "dense_manifest_path": path_to_str(dense_manifest_path),
        "loro_split_dir": path_to_str(loro_split_dir),
        "output_dir": path_to_str(output_dir),
        "mapping_csv_path": path_to_str(mapping_csv_path),
        "split_summary_csv_path": path_to_str(split_summary_csv_path),
        "summary_json_path": path_to_str(summary_json_path),
        "report_md_path": path_to_str(report_md_path),
        "error_count": len(all_errors),
        "warning_count": len(all_warnings),
        "errors": all_errors,
        "warnings": all_warnings,
        "split_summary": split_summary_df.to_dict(orient="records"),
    }

    write_json(summary_json_path, payload, overwrite=args.overwrite)

    write_markdown_report(
        path=report_md_path,
        payload=payload,
        split_summary_df=split_summary_df,
        errors=all_errors,
        warnings=all_warnings,
        overwrite=args.overwrite,
    )

    log("INFO", f"Split summary CSV: {path_to_str(split_summary_csv_path)}")
    log("INFO", f"Summary JSON:      {path_to_str(summary_json_path)}")
    log("INFO", f"Report MD:         {path_to_str(report_md_path)}")

    if status == "passed":
        log("OK", "Dense features and LORO splits are aligned. Ready for UPerNet dataset/training script.")
    else:
        log("ERROR", f"Validation failed with {len(all_errors)} errors.")
        for error in all_errors:
            log("ERROR", error)

        if args.fail_on_error:
            raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dense CROMA features against LORO split manifests."
    )

    parser.add_argument(
        "--instance-root",
        required=True,
        help="Dataset instance root.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=112,
        help="Patch stride.",
    )
    parser.add_argument(
        "--edge-mode",
        default="cover",
        help="Edge mode.",
    )
    parser.add_argument(
        "--target-modality",
        default="s2_s1_snap_vv_vh",
        help="Target modality.",
    )
    parser.add_argument(
        "--feature-key",
        default="joint_encodings",
        help="Dense CROMA feature key.",
    )
    parser.add_argument(
        "--output-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Dense feature storage dtype.",
    )

    parser.add_argument(
        "--dense-feature-path",
        default=None,
        help="Optional explicit dense .npy path.",
    )
    parser.add_argument(
        "--dense-metadata-path",
        default=None,
        help="Optional explicit dense metadata .npz path.",
    )
    parser.add_argument(
        "--dense-manifest-path",
        default=None,
        help="Optional explicit dense manifest CSV path.",
    )
    parser.add_argument(
        "--loro-split-dir",
        default=None,
        help="Optional explicit LORO split directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit validation output directory.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite validation outputs.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        default=True,
        help="Exit with code 2 if validation errors are found. Default: enabled.",
    )
    parser.add_argument(
        "--no-fail-on-error",
        dest="fail_on_error",
        action="store_false",
        help="Write reports but do not exit with error code if validation fails.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_validation(parse_args())