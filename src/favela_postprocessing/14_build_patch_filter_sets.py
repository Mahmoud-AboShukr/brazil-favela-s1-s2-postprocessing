#!/usr/bin/env python3
"""
Build experiment-ready patch filter sets from patch metadata.

Purpose
-------
This script reads patch metadata produced by:

    13_compute_patch_metadata.py

and creates controlled patch lists for downstream ML experiments.

It does NOT export image arrays and does NOT create H5 files.

Instead, it creates CSV patch lists such as:

    F01_all_patches
    F02_quality_pass
    F03_quality_positive_only
    F04_quality_pos_plus_neg_1_1_train_eval_all
    F05_quality_pos_plus_neg_1_2_train_eval_all
    F06_quality_pos_plus_neg_1_3_train_eval_all
    F07_quality_pos_gt_0_5_plus_neg_1_2_train_eval_all
    F08_quality_pos_gt_1_0_plus_neg_1_2_train_eval_all
    F09_strict_nodata5_pos_plus_neg_1_2_train_eval_all

Why this matters
----------------
The raw patch metadata contains all candidate patches. However, ML experiments usually
need controlled subsets to study:

    - all patches vs quality-filtered patches
    - positive-only training
    - positive + sampled negative training
    - different negative ratios
    - stricter positive thresholds
    - stricter nodata thresholds

Important design choice
-----------------------
For train-balanced filter sets, negative sampling is applied only to the TRAIN split.

Validation and test splits are kept as all quality-pass patches by default. This avoids
artificially balancing evaluation data, which would distort reported performance.

Inputs
------
Default patch metadata:

    <output_root>/metadata/patch_metadata_train_covered_region_test_ps512_st512_cover.csv

Outputs
-------
Output directory:

    <output_root>/metadata/patch_filter_sets_train_covered_region_test_ps512_st512_cover/

Inside it:
    filter_set_F01_all_patches.csv
    filter_set_F02_quality_pass.csv
    ...
    patch_filter_set_membership.csv
    patch_filter_sets_summary.csv
    patch_filter_sets_summary.md

Example
-------
    python3 src/favela_postprocessing/14_build_patch_filter_sets.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


SCRIPT_NAME = "14_build_patch_filter_sets.py"

DEFAULT_SPLIT_STRATEGY = "train_covered_region_test"
DEFAULT_PATCH_SIZE = 512
DEFAULT_STRIDE = 512
DEFAULT_EDGE_MODE = "cover"


@dataclass(frozen=True)
class FilterSpec:
    filter_set_id: str
    description: str
    quality_mode: str
    train_mode: str
    eval_mode: str
    train_positive_threshold: float = 0.0
    train_negative_ratio: Optional[float] = None
    recommended: bool = False


FILTER_SPECS: List[FilterSpec] = [
    FilterSpec(
        filter_set_id="F01_all_patches",
        description="All patches without quality filtering.",
        quality_mode="all",
        train_mode="all",
        eval_mode="same_as_train",
        recommended=False,
    ),
    FilterSpec(
        filter_set_id="F02_quality_pass",
        description="All patches that pass the basic quality filter.",
        quality_mode="basic_quality",
        train_mode="all",
        eval_mode="same_as_train",
        recommended=True,
    ),
    FilterSpec(
        filter_set_id="F03_quality_positive_only",
        description="Only positive patches that pass the basic quality filter.",
        quality_mode="basic_quality",
        train_mode="positive_only",
        eval_mode="same_as_train",
        train_positive_threshold=0.0,
        recommended=False,
    ),
    FilterSpec(
        filter_set_id="F04_quality_pos_plus_neg_1_1_train_eval_all",
        description=(
            "Training uses all quality positive patches plus sampled quality negatives at 1:1. "
            "Validation/test keep all quality patches."
        ),
        quality_mode="basic_quality",
        train_mode="positive_plus_sampled_negative",
        eval_mode="quality_all",
        train_positive_threshold=0.0,
        train_negative_ratio=1.0,
        recommended=False,
    ),
    FilterSpec(
        filter_set_id="F05_quality_pos_plus_neg_1_2_train_eval_all",
        description=(
            "Training uses all quality positive patches plus sampled quality negatives at 1:2. "
            "Validation/test keep all quality patches."
        ),
        quality_mode="basic_quality",
        train_mode="positive_plus_sampled_negative",
        eval_mode="quality_all",
        train_positive_threshold=0.0,
        train_negative_ratio=2.0,
        recommended=True,
    ),
    FilterSpec(
        filter_set_id="F06_quality_pos_plus_neg_1_3_train_eval_all",
        description=(
            "Training uses all quality positive patches plus sampled quality negatives at 1:3. "
            "Validation/test keep all quality patches."
        ),
        quality_mode="basic_quality",
        train_mode="positive_plus_sampled_negative",
        eval_mode="quality_all",
        train_positive_threshold=0.0,
        train_negative_ratio=3.0,
        recommended=False,
    ),
    FilterSpec(
        filter_set_id="F07_quality_pos_gt_0_5_plus_neg_1_2_train_eval_all",
        description=(
            "Training uses quality positive patches with label_positive_percent > 0.5 plus "
            "sampled quality negatives at 1:2. Validation/test keep all quality patches."
        ),
        quality_mode="basic_quality",
        train_mode="positive_plus_sampled_negative",
        eval_mode="quality_all",
        train_positive_threshold=0.5,
        train_negative_ratio=2.0,
        recommended=False,
    ),
    FilterSpec(
        filter_set_id="F08_quality_pos_gt_1_0_plus_neg_1_2_train_eval_all",
        description=(
            "Training uses quality positive patches with label_positive_percent > 1.0 plus "
            "sampled quality negatives at 1:2. Validation/test keep all quality patches."
        ),
        quality_mode="basic_quality",
        train_mode="positive_plus_sampled_negative",
        eval_mode="quality_all",
        train_positive_threshold=1.0,
        train_negative_ratio=2.0,
        recommended=False,
    ),
    FilterSpec(
        filter_set_id="F09_strict_nodata5_pos_plus_neg_1_2_train_eval_all",
        description=(
            "Strict subset: patches must pass basic quality and have max_nodata_percent <= 5. "
            "Training uses positives plus sampled negatives at 1:2. Validation/test keep all "
            "strict-quality patches."
        ),
        quality_mode="strict_nodata5",
        train_mode="positive_plus_sampled_negative",
        eval_mode="quality_all",
        train_positive_threshold=0.0,
        train_negative_ratio=2.0,
        recommended=False,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build controlled patch filter sets for ML experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--patch-metadata",
        type=Path,
        default=None,
        help=(
            "Explicit patch metadata CSV. If omitted, uses the default "
            "train_covered_region_test ps512 st512 cover file."
        ),
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default=DEFAULT_SPLIT_STRATEGY,
        help=f"Split strategy used for default metadata lookup. Default: {DEFAULT_SPLIT_STRATEGY}",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Patch size used for default metadata lookup. Default: {DEFAULT_PATCH_SIZE}",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Stride used for default metadata lookup. Default: {DEFAULT_STRIDE}",
    )
    parser.add_argument(
        "--edge-mode",
        type=str,
        default=DEFAULT_EDGE_MODE,
        help=f"Edge mode used for default metadata lookup. Default: {DEFAULT_EDGE_MODE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic negative sampling. Default: 42",
    )
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="Include only one city. Can be repeated.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory contents if files already exist.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "output_root" not in cfg:
        raise KeyError("Missing required key in config: output_root")

    return cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_city_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def bool_from_any(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def default_patch_metadata_path(
    output_root: Path,
    split_strategy: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
) -> Path:
    suffix = f"{split_strategy}_ps{patch_size}_st{stride}_{edge_mode}"
    return output_root / "metadata" / f"patch_metadata_{suffix}.csv"


def suffix_from_patch_metadata(path: Path) -> str:
    name = path.stem
    prefix = "patch_metadata_"

    if name.startswith(prefix):
        return name[len(prefix):]

    return name


def load_patch_metadata(
    path: Path,
    selected_cities: Optional[Sequence[str]],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Patch metadata CSV not found: {path}\n"
            f"Run 13_compute_patch_metadata.py first."
        )

    df = pd.read_csv(path)

    required = [
        "patch_id",
        "city",
        "split_strategy",
        "fold_id",
        "split",
        "region",
        "label_positive_percent",
        "is_positive_patch",
        "passes_basic_quality_filter",
        "max_nodata_percent",
        "s2_nodata_percent",
        "s1_nodata_percent",
        "label_nodata_percent",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Patch metadata is missing required columns: {missing}")

    df = df.copy()
    df["city"] = df["city"].map(normalize_city_name)

    bool_cols = [
        "is_positive_patch",
        "passes_basic_quality_filter",
        "passes_nodata_filter",
        "passes_cloud_filter",
        "positive_gt_0pct",
        "positive_gt_0_5pct",
        "positive_gt_1pct",
        "positive_gt_2pct",
    ]

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(bool_from_any)

    if selected_cities:
        selected = {normalize_city_name(city) for city in selected_cities}
        df = df[df["city"].isin(selected)].copy()

    if df.empty:
        raise RuntimeError("No patch metadata rows selected.")

    return df.reset_index(drop=True)


def stable_random_state(seed: int, *parts: Any) -> int:
    text = "|".join([str(seed)] + [str(part) for part in parts])
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def quality_mask(df: pd.DataFrame, quality_mode: str) -> pd.Series:
    if quality_mode == "all":
        return pd.Series(True, index=df.index)

    if quality_mode == "basic_quality":
        return df["passes_basic_quality_filter"].map(bool_from_any)

    if quality_mode == "strict_nodata5":
        return (
            df["passes_basic_quality_filter"].map(bool_from_any)
            & (df["max_nodata_percent"].astype(float) <= 5.0)
            & (df["s2_nodata_percent"].astype(float) <= 5.0)
            & (df["s1_nodata_percent"].astype(float) <= 5.0)
            & (df["label_nodata_percent"].astype(float) <= 5.0)
        )

    raise ValueError(f"Unknown quality_mode: {quality_mode}")


def positive_mask(df: pd.DataFrame, threshold: float) -> pd.Series:
    return df["label_positive_percent"].astype(float) > float(threshold)


def add_filter_columns(
    df: pd.DataFrame,
    spec: FilterSpec,
    inclusion_group: str,
) -> pd.DataFrame:
    out = df.copy()

    out.insert(0, "filter_set_id", spec.filter_set_id)
    out.insert(1, "filter_set_description", spec.description)
    out.insert(2, "filter_inclusion_group", inclusion_group)

    out["filter_quality_mode"] = spec.quality_mode
    out["filter_train_mode"] = spec.train_mode
    out["filter_eval_mode"] = spec.eval_mode
    out["filter_train_positive_threshold"] = spec.train_positive_threshold
    out["filter_train_negative_ratio"] = (
        math.nan if spec.train_negative_ratio is None else spec.train_negative_ratio
    )
    out["filter_recommended"] = spec.recommended

    return out


def build_all_or_positive_filter(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    qmask = quality_mask(df, spec.quality_mode)

    if spec.train_mode == "all":
        selected = df[qmask].copy()
        return add_filter_columns(selected, spec, "all_selected")

    if spec.train_mode == "positive_only":
        pmask = positive_mask(df, spec.train_positive_threshold)
        selected = df[qmask & pmask].copy()
        return add_filter_columns(selected, spec, "positive_only")

    raise ValueError(f"Unexpected train_mode for simple filter: {spec.train_mode}")


def build_train_balanced_eval_all_filter(
    df: pd.DataFrame,
    spec: FilterSpec,
    seed: int,
) -> pd.DataFrame:
    if spec.train_negative_ratio is None:
        raise ValueError(f"Filter {spec.filter_set_id} requires train_negative_ratio.")

    qmask = quality_mask(df, spec.quality_mode)

    train_mask = df["split"].astype(str) == "train"
    eval_mask = ~train_mask

    train_df = df[qmask & train_mask].copy()
    eval_df = df[qmask & eval_mask].copy()

    selected_train_parts: List[pd.DataFrame] = []

    group_cols = ["split_strategy", "fold_id"]

    for group_key, group in train_df.groupby(group_cols, dropna=False):
        positives = group[positive_mask(group, spec.train_positive_threshold)].copy()
        negatives = group[~positive_mask(group, spec.train_positive_threshold)].copy()

        n_pos = len(positives)
        n_neg_target = int(round(n_pos * float(spec.train_negative_ratio)))
        n_neg_sample = min(len(negatives), n_neg_target)

        if n_pos > 0:
            positives = add_filter_columns(
                positives,
                spec,
                "train_positive",
            )
            selected_train_parts.append(positives)

        if n_neg_sample > 0:
            random_state = stable_random_state(seed, spec.filter_set_id, group_key, "negative")
            sampled_negatives = negatives.sample(
                n=n_neg_sample,
                random_state=random_state,
                replace=False,
            ).copy()

            sampled_negatives = add_filter_columns(
                sampled_negatives,
                spec,
                "train_negative_sampled",
            )
            selected_train_parts.append(sampled_negatives)

    if spec.eval_mode == "quality_all":
        eval_selected = add_filter_columns(
            eval_df,
            spec,
            "eval_quality_all",
        )
    elif spec.eval_mode == "same_as_train":
        eval_selected = add_filter_columns(
            eval_df[positive_mask(eval_df, spec.train_positive_threshold)].copy(),
            spec,
            "eval_positive_only",
        )
    else:
        raise ValueError(f"Unknown eval_mode: {spec.eval_mode}")

    parts = selected_train_parts + [eval_selected]

    if not parts:
        return add_filter_columns(df.iloc[0:0].copy(), spec, "empty")

    out = pd.concat(parts, ignore_index=True)

    return out


def build_filter_set(
    df: pd.DataFrame,
    spec: FilterSpec,
    seed: int,
) -> pd.DataFrame:
    if spec.train_mode in {"all", "positive_only"}:
        return build_all_or_positive_filter(df, spec)

    if spec.train_mode == "positive_plus_sampled_negative":
        return build_train_balanced_eval_all_filter(df, spec, seed)

    raise ValueError(f"Unknown train_mode: {spec.train_mode}")


def summarize_filter_set(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(
            [
                "filter_set_id",
                "filter_inclusion_group",
                "split_strategy",
                "fold_id",
                "heldout_region",
                "split",
                "region",
            ],
            dropna=False,
        )
        .agg(
            patch_count=("patch_id", "count"),
            city_count=("city", "nunique"),
            positive_patch_count=("is_positive_patch", "sum"),
            mean_label_positive_percent=("label_positive_percent", "mean"),
            mean_max_nodata_percent=("max_nodata_percent", "mean"),
            quality_pass_count=("passes_basic_quality_filter", "sum"),
        )
        .reset_index()
        .sort_values(
            [
                "filter_set_id",
                "split_strategy",
                "fold_id",
                "split",
                "region",
                "filter_inclusion_group",
            ]
        )
    )

    summary["negative_patch_count"] = summary["patch_count"] - summary["positive_patch_count"]
    summary["positive_patch_percent"] = (
        100.0 * summary["positive_patch_count"] / summary["patch_count"]
    )

    return summary


def summarize_filter_set_overall(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for filter_set_id, sub in df.groupby("filter_set_id", dropna=False):
        spec_description = str(sub["filter_set_description"].iloc[0])
        recommended = bool_from_any(sub["filter_recommended"].iloc[0])

        row = {
            "filter_set_id": filter_set_id,
            "recommended": recommended,
            "description": spec_description,
            "total_patches": len(sub),
            "train_patches": int((sub["split"] == "train").sum()),
            "val_patches": int((sub["split"] == "val").sum()),
            "test_patches": int((sub["split"] == "test").sum()),
            "positive_patches": int(sub["is_positive_patch"].sum()),
            "negative_patches": int((~sub["is_positive_patch"].map(bool_from_any)).sum()),
            "positive_patch_percent": 100.0 * int(sub["is_positive_patch"].sum()) / len(sub),
            "cities": int(sub["city"].nunique()),
            "folds": int(sub["fold_id"].nunique()),
            "mean_label_positive_percent": float(sub["label_positive_percent"].mean()),
            "mean_max_nodata_percent": float(sub["max_nodata_percent"].mean()),
        }

        train = sub[sub["split"] == "train"].copy()

        if len(train) > 0:
            train_pos = int(train["is_positive_patch"].sum())
            train_neg = int((~train["is_positive_patch"].map(bool_from_any)).sum())
            row["train_positive_patches"] = train_pos
            row["train_negative_patches"] = train_neg
            row["train_negative_to_positive_ratio"] = (
                train_neg / train_pos if train_pos > 0 else math.nan
            )
        else:
            row["train_positive_patches"] = 0
            row["train_negative_patches"] = 0
            row["train_negative_to_positive_ratio"] = math.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("filter_set_id").reset_index(drop=True)


def markdown_escape(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return "nan"
    except Exception:
        pass

    text = str(value)
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


def df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    columns = list(df.columns)

    header = "| " + " | ".join(markdown_escape(col) for col in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    rows = []

    for _, row in df.iterrows():
        rows.append(
            "| "
            + " | ".join(markdown_escape(row[col]) for col in columns)
            + " |"
        )

    return "\n".join([header, separator] + rows)


def write_markdown_summary(
    output_path: Path,
    metadata_path: Path,
    overall_summary: pd.DataFrame,
    detailed_summary: pd.DataFrame,
    filter_specs: Sequence[FilterSpec],
) -> None:
    ensure_dir(output_path.parent)

    lines: List[str] = []

    lines.append("# Patch Filter Sets Summary")
    lines.append("")
    lines.append(f"Generated by `{SCRIPT_NAME}`.")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "This report summarizes experiment-ready patch lists generated from patch metadata. "
        "The goal is to support controlled ML experiments without duplicating image arrays "
        "or exporting H5 files."
    )
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append(f"- Patch metadata: `{metadata_path}`")
    lines.append("")
    lines.append("## Filter-set definitions")
    lines.append("")

    spec_rows = []

    for spec in filter_specs:
        spec_rows.append(
            {
                "filter_set_id": spec.filter_set_id,
                "recommended": spec.recommended,
                "quality_mode": spec.quality_mode,
                "train_mode": spec.train_mode,
                "eval_mode": spec.eval_mode,
                "train_positive_threshold": spec.train_positive_threshold,
                "train_negative_ratio": spec.train_negative_ratio,
                "description": spec.description,
            }
        )

    lines.append(df_to_markdown_table(pd.DataFrame(spec_rows)))
    lines.append("")

    lines.append("## Overall summary")
    lines.append("")

    display_overall = overall_summary.copy()

    for col in [
        "positive_patch_percent",
        "mean_label_positive_percent",
        "mean_max_nodata_percent",
        "train_negative_to_positive_ratio",
    ]:
        if col in display_overall.columns:
            display_overall[col] = display_overall[col].map(
                lambda x: "nan" if pd.isna(x) else f"{float(x):.3f}"
            )

    lines.append(df_to_markdown_table(display_overall))
    lines.append("")

    lines.append("## Detailed summary by split and region")
    lines.append("")

    display_cols = [
        "filter_set_id",
        "filter_inclusion_group",
        "split",
        "region",
        "patch_count",
        "city_count",
        "positive_patch_count",
        "negative_patch_count",
        "positive_patch_percent",
    ]

    existing_cols = [col for col in display_cols if col in detailed_summary.columns]
    display_detail = detailed_summary[existing_cols].copy()

    if "positive_patch_percent" in display_detail.columns:
        display_detail["positive_patch_percent"] = display_detail["positive_patch_percent"].map(
            lambda x: f"{float(x):.3f}"
        )

    lines.append(df_to_markdown_table(display_detail))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `F01_all_patches` is useful for diagnostics but includes low-quality patches.")
    lines.append("- `F02_quality_pass` is the cleanest direct subset of the patch metadata.")
    lines.append("- `F05_quality_pos_plus_neg_1_2_train_eval_all` is a good first training subset because it keeps all quality positives and samples negatives at a moderate ratio.")
    lines.append("- Validation and test patches are not artificially class-balanced in train-balanced filter sets; they remain quality-filtered evaluation sets.")
    lines.append("- These CSVs are patch lists. The dataloader should still read pixels directly from GeoTIFFs using the window columns.")
    lines.append("")

    lines.append("## Recommended next step")
    lines.append("")
    lines.append(
        "Use one recommended filter set, usually `F05_quality_pos_plus_neg_1_2_train_eval_all`, "
        "to compute training-only normalization statistics and then build a PyTorch GeoTIFF dataloader."
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def output_directory(output_root: Path, suffix: str) -> Path:
    return output_root / "metadata" / f"patch_filter_sets_{suffix}"


def write_filter_set_csv(df: pd.DataFrame, out_dir: Path, filter_set_id: str, overwrite: bool) -> Path:
    path = out_dir / f"filter_set_{filter_set_id}.csv"

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}\n"
            f"Use --overwrite to replace it."
        )

    df.to_csv(path, index=False)
    return path


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))

    if args.patch_metadata is None:
        patch_metadata_path = default_patch_metadata_path(
            output_root=output_root,
            split_strategy=args.split_strategy,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
        )
    else:
        patch_metadata_path = args.patch_metadata

    suffix = suffix_from_patch_metadata(patch_metadata_path)
    out_dir = output_directory(output_root, suffix)
    ensure_dir(out_dir)

    print("[INFO] Build patch filter sets")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Patch metadata: {patch_metadata_path}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Seed: {args.seed}")

    df = load_patch_metadata(
        path=patch_metadata_path,
        selected_cities=args.city,
    )

    print(f"[INFO] Patch rows loaded: {len(df)}")
    print(f"[INFO] Cities: {df['city'].nunique()}")
    print(f"[INFO] Splits: {sorted(df['split'].astype(str).unique().tolist())}")
    print(f"[INFO] Positive patches: {int(df['is_positive_patch'].sum())}")
    print(f"[INFO] Quality-pass patches: {int(df['passes_basic_quality_filter'].sum())}")

    filter_set_frames: List[pd.DataFrame] = []
    written_files: List[Path] = []

    for spec in FILTER_SPECS:
        selected = build_filter_set(
            df=df,
            spec=spec,
            seed=args.seed,
        )

        selected = selected.sort_values(
            [
                "filter_set_id",
                "split_strategy",
                "fold_id",
                "split",
                "region_order",
                "city",
                "row_start",
                "col_start",
            ]
        ).reset_index(drop=True)

        out_path = write_filter_set_csv(
            df=selected,
            out_dir=out_dir,
            filter_set_id=spec.filter_set_id,
            overwrite=args.overwrite,
        )

        written_files.append(out_path)
        filter_set_frames.append(selected)

        train = selected[selected["split"] == "train"]

        if len(train) > 0:
            train_pos = int(train["is_positive_patch"].sum())
            train_neg = int((~train["is_positive_patch"].map(bool_from_any)).sum())
            ratio = train_neg / train_pos if train_pos > 0 else math.nan
        else:
            train_pos = 0
            train_neg = 0
            ratio = math.nan

        print(
            f"[INFO] {spec.filter_set_id}: "
            f"total={len(selected)}, "
            f"train={len(train)}, "
            f"train_pos={train_pos}, "
            f"train_neg={train_neg}, "
            f"train_neg_pos_ratio={ratio:.3f}" if not pd.isna(ratio) else
            f"[INFO] {spec.filter_set_id}: total={len(selected)}, train={len(train)}, ratio=nan"
        )

    combined = pd.concat(filter_set_frames, ignore_index=True)

    membership_cols = [
        "filter_set_id",
        "patch_id",
        "filter_inclusion_group",
        "split_strategy",
        "fold_id",
        "heldout_region",
        "split",
        "city",
        "region",
        "is_positive_patch",
        "label_positive_percent",
        "passes_basic_quality_filter",
        "max_nodata_percent",
    ]

    existing_membership_cols = [col for col in membership_cols if col in combined.columns]

    membership = combined[existing_membership_cols].copy()
    membership_path = out_dir / "patch_filter_set_membership.csv"
    membership.to_csv(membership_path, index=False)

    detailed_summary = summarize_filter_set(combined)
    detailed_summary_path = out_dir / "patch_filter_sets_detailed_summary.csv"
    detailed_summary.to_csv(detailed_summary_path, index=False)

    overall_summary = summarize_filter_set_overall(combined)
    overall_summary_path = out_dir / "patch_filter_sets_summary.csv"
    overall_summary.to_csv(overall_summary_path, index=False)

    md_path = out_dir / "patch_filter_sets_summary.md"

    write_markdown_summary(
        output_path=md_path,
        metadata_path=patch_metadata_path,
        overall_summary=overall_summary,
        detailed_summary=detailed_summary,
        filter_specs=FILTER_SPECS,
    )

    print(f"[INFO] Wrote membership CSV: {membership_path}")
    print(f"[INFO] Wrote detailed summary CSV: {detailed_summary_path}")
    print(f"[INFO] Wrote overall summary CSV: {overall_summary_path}")
    print(f"[INFO] Wrote Markdown summary: {md_path}")

    print("[INFO] Written filter-set files:")
    for path in written_files:
        print(f"       - {path}")

    print("[INFO] Patch filter sets built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())