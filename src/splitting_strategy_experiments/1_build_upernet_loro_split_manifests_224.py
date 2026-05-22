"""
15_build_upernet_loro_split_manifests_224.py

Main objective
--------------
Build Leave-One-Region-Out (LORO) train/validation/test split manifests
for the UPerNet/CROMA segmentation experiment.

This script:
1. Reads the CROMA comparison manifest.
2. Keeps only the target modality: S2 + SNAP-GRD VV/VH by default.
3. Deduplicates to one row per unique spatial patch.
4. Builds one LORO fold per Brazilian region.
5. Uses the held-out region as test.
6. Selects validation cities from the remaining regions.
7. Guarantees validation is city-disjoint from training.
8. Writes train/val/test CSV files for each fold.
9. Writes summary files in CSV, JSON, and Markdown.
10. Prints clear logs and warnings.

Recommended use
---------------
python src/upernet_croma/15_build_upernet_loro_split_manifests_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --target-modality "s2_s1_snap_vv_vh" `
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EXPECTED_REGIONS = [
    "Central-West",
    "North",
    "Northeast",
    "South",
    "Southeast",
]


CITY_COLUMN_CANDIDATES = [
    "city",
    "city_name",
    "municipality",
    "municipio",
    "mun_name",
    "name_city",
]

REGION_COLUMN_CANDIDATES = [
    "region",
    "brazil_region",
    "macroregion",
    "macro_region",
    "geographic_region",
]

MODALITY_COLUMN_CANDIDATES = [
    "modality",
    "input_modality",
    "mode",
    "data_modality",
    "source_modality",
]

PATCH_ID_COLUMN_CANDIDATES = [
    "patch_id",
    "unique_patch_id",
    "patch_uid",
    "sample_id",
    "window_id",
    "tile_id",
    "id",
]

X_COLUMN_CANDIDATES = [
    "x",
    "x0",
    "x_off",
    "x_offset",
    "col",
    "col_off",
    "column",
    "patch_col",
    "window_col",
    "window_col_off",
    "left",
]

Y_COLUMN_CANDIDATES = [
    "y",
    "y0",
    "y_off",
    "y_offset",
    "row",
    "row_off",
    "patch_row",
    "window_row",
    "window_row_off",
    "top",
]

POSITIVE_PATCH_COLUMN_CANDIDATES = [
    "is_positive",
    "positive",
    "has_favela",
    "contains_favela",
    "patch_positive",
    "label_positive",
    "target",
    "label",
    "y",
]

POSITIVE_PIXEL_COLUMN_CANDIDATES = [
    "favela_pixels",
    "positive_pixels",
    "pos_pixels",
    "label_positive_pixels",
    "n_positive_pixels",
    "n_favela_pixels",
    "label_sum",
    "mask_sum",
    "sum_label",
    "favela_pixel_count",
]

TOTAL_PIXEL_COLUMN_CANDIDATES = [
    "total_pixels",
    "patch_pixels",
    "n_pixels",
    "num_pixels",
    "valid_pixels",
    "label_total_pixels",
    "mask_total_pixels",
]

FAVELA_PCT_COLUMN_CANDIDATES = [
    "favela_pct",
    "favela_percent",
    "favela_percentage",
    "favela_pixel_pct",
    "favela_pixel_percentage",
    "favela_coverage",
    "favela_coverage_pct",
    "favela_coverage_percent",
    "label_positive_pct",
    "label_positive_percent",
    "positive_pixel_pct",
    "positive_pixel_percentage",
    "positive_fraction",
    "label_fraction",
    "mask_fraction",
    "coverage_pct",
    "coverage_percent",
]


def log(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"[WARNING] {message}", flush=True)


def fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def normalize_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    explicit: Optional[str] = None,
    required: bool = False,
    purpose: str = "column",
) -> Optional[str]:
    if explicit:
        if explicit in df.columns:
            return explicit
        fail(
            f"Requested {purpose} '{explicit}' was not found. "
            f"Available columns are: {list(df.columns)}"
        )

    norm_to_original = {normalize_name(c): c for c in df.columns}

    for candidate in candidates:
        norm = normalize_name(candidate)
        if norm in norm_to_original:
            return norm_to_original[norm]

    if required:
        fail(
            f"Could not automatically detect {purpose}. "
            f"Tried candidates: {candidates}. "
            f"Available columns are: {list(df.columns)}"
        )

    return None


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.json")) + list(output_dir.glob("*.md"))
    if existing_files and not overwrite:
        fail(
            f"Output directory already contains split files:\n"
            f"{output_dir}\n\n"
            f"Use --overwrite if you want to replace them."
        )


def default_manifest_path(instance_root: Path, patch_size: int, stride: int, edge_mode: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / f"croma_comparison_manifest_ps{patch_size}_st{stride}_{edge_mode}.csv"
    )


def default_output_dir(instance_root: Path, patch_size: int, stride: int, edge_mode: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "upernet_croma"
        / f"splits_loro_ps{patch_size}_st{stride}_{edge_mode}"
    )


def slug_region(region: str) -> str:
    text = str(region).strip()
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Manifest file does not exist:\n{path}")

    log(f"[1/8] Reading manifest:\n{path}")
    df = pd.read_csv(path)

    if df.empty:
        fail("Manifest is empty.")

    log(f"      Loaded rows: {len(df):,}")
    log(f"      Loaded columns: {len(df.columns):,}")

    return df


def filter_target_modality(
    df: pd.DataFrame,
    modality_col: Optional[str],
    target_modality: str,
) -> pd.DataFrame:
    if modality_col is None:
        warn(
            "No modality column was detected. "
            "The script will deduplicate the full manifest, but cannot explicitly filter to the target modality."
        )
        return df.copy()

    log(f"[2/8] Filtering target modality using column '{modality_col}'")
    values = df[modality_col].astype(str)
    norm_values = values.map(normalize_value)
    target_norm = normalize_value(target_modality)

    exact_mask = norm_values == target_norm

    if exact_mask.any():
        out = df.loc[exact_mask].copy()
        log(f"      Target modality: {target_modality}")
        log(f"      Rows after modality filtering: {len(out):,}")
        return out

    target_tokens = [tok for tok in target_norm.split("_") if tok]
    token_mask = norm_values.apply(lambda x: all(tok in x for tok in target_tokens))

    if token_mask.any():
        out = df.loc[token_mask].copy()
        log(f"      Target modality: {target_modality}")
        log("      Exact modality match was not found, but token-based matching succeeded.")
        log(f"      Rows after modality filtering: {len(out):,}")
        return out

    unique_modalities = sorted(df[modality_col].dropna().astype(str).unique().tolist())
    fail(
        f"Could not find target modality '{target_modality}' in column '{modality_col}'.\n"
        f"Available modality values are:\n{unique_modalities}"
    )


def build_patch_id(
    df: pd.DataFrame,
    city_col: str,
    region_col: str,
    explicit_patch_id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    patch_col = find_column(
        df,
        PATCH_ID_COLUMN_CANDIDATES,
        explicit=explicit_patch_id_col,
        required=False,
        purpose="patch id column",
    )

    out = df.copy()

    if patch_col is not None:
        log(f"[3/8] Using existing patch id column: '{patch_col}'")
        out["__patch_id"] = out[patch_col].astype(str)
        return out, "__patch_id"

    x_col = find_column(df, X_COLUMN_CANDIDATES, required=False, purpose="patch x/column coordinate")
    y_col = find_column(df, Y_COLUMN_CANDIDATES, required=False, purpose="patch y/row coordinate")

    if x_col is not None and y_col is not None:
        log(
            "[3/8] No patch id column found. "
            f"Building patch id from city, region, '{x_col}', and '{y_col}'."
        )
        out["__patch_id"] = (
            out[region_col].astype(str)
            + "__"
            + out[city_col].astype(str)
            + "__x"
            + out[x_col].astype(str)
            + "__y"
            + out[y_col].astype(str)
        )
        return out, "__patch_id"

    fail(
        "Could not identify a patch id column or x/y patch coordinates.\n"
        "Please rerun with --patch-id-col <COLUMN_NAME>.\n"
        f"Available columns are: {list(df.columns)}"
    )


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infer_positive_column(
    df: pd.DataFrame,
    explicit_positive_col: Optional[str],
    positive_threshold: float,
) -> Tuple[pd.DataFrame, str, Optional[str], Optional[str], Optional[str]]:
    """
    Returns:
        df with __positive_patch
        positive source description
        positive pixel column
        total pixel column
        favela pct column
    """

    out = df.copy()

    positive_col = find_column(
        out,
        POSITIVE_PATCH_COLUMN_CANDIDATES,
        explicit=explicit_positive_col,
        required=False,
        purpose="positive patch column",
    )

    pos_pixel_col = find_column(
        out,
        POSITIVE_PIXEL_COLUMN_CANDIDATES,
        required=False,
        purpose="positive pixel count column",
    )

    total_pixel_col = find_column(
        out,
        TOTAL_PIXEL_COLUMN_CANDIDATES,
        required=False,
        purpose="total pixel count column",
    )

    favela_pct_col = find_column(
        out,
        FAVELA_PCT_COLUMN_CANDIDATES,
        required=False,
        purpose="favela percentage/fraction column",
    )

    if positive_col is not None:
        vals_raw = out[positive_col]
        vals_num = numeric_series(vals_raw)

        if vals_num.notna().sum() > 0:
            out["__positive_patch"] = vals_num.fillna(0) > positive_threshold
        else:
            text_vals = vals_raw.astype(str).str.lower().str.strip()
            out["__positive_patch"] = text_vals.isin(["true", "yes", "y", "positive", "pos", "1"])

        return out, f"positive patch column: {positive_col}", pos_pixel_col, total_pixel_col, favela_pct_col

    if pos_pixel_col is not None:
        vals = numeric_series(out[pos_pixel_col]).fillna(0)
        out["__positive_patch"] = vals > positive_threshold
        return out, f"positive pixel column: {pos_pixel_col}", pos_pixel_col, total_pixel_col, favela_pct_col

    if favela_pct_col is not None:
        vals = numeric_series(out[favela_pct_col]).fillna(0)
        out["__positive_patch"] = vals > positive_threshold
        return out, f"favela percentage column: {favela_pct_col}", pos_pixel_col, total_pixel_col, favela_pct_col

    fail(
        "Could not infer whether a patch is favela-positive.\n"
        "Please provide --positive-col <COLUMN_NAME>.\n"
        f"Available columns are: {list(out.columns)}"
    )


def deduplicate_patches(
    df: pd.DataFrame,
    patch_id_col: str,
) -> pd.DataFrame:
    log("[4/8] Deduplicating to one row per unique spatial patch")

    before = len(df)
    duplicated = df.duplicated(subset=[patch_id_col], keep=False).sum()

    if duplicated > 0:
        warn(
            f"Found {duplicated:,} rows with duplicated patch ids after modality filtering. "
            "Keeping the first row for each patch id."
        )

    out = df.drop_duplicates(subset=[patch_id_col], keep="first").copy()
    after = len(out)

    log(f"      Rows before deduplication: {before:,}")
    log(f"      Unique patches after deduplication: {after:,}")

    if after == 0:
        fail("No patches remain after deduplication.")

    return out


def compute_favela_pixel_pct(
    df: pd.DataFrame,
    pos_pixel_col: Optional[str],
    total_pixel_col: Optional[str],
    favela_pct_col: Optional[str],
) -> float:
    if pos_pixel_col is not None and total_pixel_col is not None:
        pos_pixels = numeric_series(df[pos_pixel_col]).fillna(0).sum()
        total_pixels = numeric_series(df[total_pixel_col]).fillna(0).sum()

        if total_pixels > 0:
            return float(100.0 * pos_pixels / total_pixels)

    if favela_pct_col is not None:
        vals = numeric_series(df[favela_pct_col]).replace([np.inf, -np.inf], np.nan).dropna()

        if len(vals) > 0:
            max_val = vals.max()
            mean_val = vals.mean()

            if max_val <= 1.5:
                return float(mean_val * 100.0)

            return float(mean_val)

    return float("nan")


def split_stats(
    df: pd.DataFrame,
    city_col: str,
    region_col: str,
    patch_id_col: str,
    pos_pixel_col: Optional[str],
    total_pixel_col: Optional[str],
    favela_pct_col: Optional[str],
) -> Dict[str, Any]:
    n = int(len(df))
    pos = int(df["__positive_patch"].sum()) if n else 0
    empty = int(n - pos)

    favela_pixel_pct = compute_favela_pixel_pct(
        df=df,
        pos_pixel_col=pos_pixel_col,
        total_pixel_col=total_pixel_col,
        favela_pct_col=favela_pct_col,
    )

    cities = sorted(df[city_col].dropna().astype(str).unique().tolist())
    regions = sorted(df[region_col].dropna().astype(str).unique().tolist())

    return {
        "patches": n,
        "positive_patches": pos,
        "empty_patches": empty,
        "positive_patch_pct": float(100.0 * pos / n) if n else float("nan"),
        "favela_pixel_pct": favela_pixel_pct,
        "n_cities": len(cities),
        "cities": cities,
        "n_regions": len(regions),
        "regions": regions,
        "unique_patch_ids": int(df[patch_id_col].nunique()) if n else 0,
    }


def make_city_stats(
    df: pd.DataFrame,
    city_col: str,
    region_col: str,
    pos_pixel_col: Optional[str],
    total_pixel_col: Optional[str],
    favela_pct_col: Optional[str],
) -> pd.DataFrame:
    rows = []

    for (region, city), g in df.groupby([region_col, city_col], dropna=False):
        n = len(g)
        pos = int(g["__positive_patch"].sum())
        empty = int(n - pos)
        favela_pixel_pct = compute_favela_pixel_pct(
            g,
            pos_pixel_col=pos_pixel_col,
            total_pixel_col=total_pixel_col,
            favela_pct_col=favela_pct_col,
        )

        rows.append(
            {
                "region": str(region),
                "city": str(city),
                "patches": int(n),
                "positive_patches": pos,
                "empty_patches": empty,
                "positive_patch_pct": float(100.0 * pos / n) if n else float("nan"),
                "favela_pixel_pct": favela_pixel_pct,
            }
        )

    city_stats = pd.DataFrame(rows)

    if city_stats.empty:
        fail("City statistics table is empty. Cannot select validation cities.")

    return city_stats.sort_values(["region", "city"]).reset_index(drop=True)


def choose_validation_cities(
    trainval_df: pd.DataFrame,
    city_col: str,
    region_col: str,
    pos_pixel_col: Optional[str],
    total_pixel_col: Optional[str],
    favela_pct_col: Optional[str],
    val_fraction: float,
    min_val_patches: int,
    min_val_positive_patches: int,
) -> Tuple[List[str], pd.DataFrame, List[str]]:
    warnings: List[str] = []

    city_stats = make_city_stats(
        trainval_df,
        city_col=city_col,
        region_col=region_col,
        pos_pixel_col=pos_pixel_col,
        total_pixel_col=total_pixel_col,
        favela_pct_col=favela_pct_col,
    )

    total_trainval_patches = len(trainval_df)
    target_val_patches = max(min_val_patches, int(round(total_trainval_patches * val_fraction)))
    target_val_patches = min(target_val_patches, max(1, total_trainval_patches - 1))

    remaining_regions = sorted(trainval_df[region_col].dropna().astype(str).unique().tolist())
    target_per_region = max(1.0, target_val_patches / max(1, len(remaining_regions)))

    selected: List[str] = []

    def selected_stats(selected_cities: Sequence[str]) -> Tuple[int, int]:
        if not selected_cities:
            return 0, 0

        sub = city_stats[city_stats["city"].isin(selected_cities)]
        return int(sub["patches"].sum()), int(sub["positive_patches"].sum())

    def region_would_keep_train_city(city: str) -> bool:
        row = city_stats.loc[city_stats["city"] == city].iloc[0]
        region = row["region"]
        region_cities = city_stats.loc[city_stats["region"] == region, "city"].tolist()
        already_selected_in_region = [c for c in selected if c in region_cities]
        return len(region_cities) - len(already_selected_in_region) > 1

    # First pass: try to pick one validation city from each remaining region,
    # while leaving at least one city from that region for training.
    for region in remaining_regions:
        candidates = city_stats[city_stats["region"] == region].copy()

        if len(candidates) <= 1:
            warnings.append(
                f"Region '{region}' has only one city in train/val pool; "
                "it was not used for validation to avoid removing the region from training."
            )
            continue

        positive_candidates = candidates[candidates["positive_patches"] > 0].copy()
        if not positive_candidates.empty:
            candidates = positive_candidates

        candidates["score"] = (
            (candidates["patches"] - target_per_region).abs()
            - 0.10 * candidates["positive_patches"].clip(upper=max(1, min_val_positive_patches))
        )

        chosen = candidates.sort_values(["score", "patches", "city"]).iloc[0]["city"]
        selected.append(str(chosen))

    # Second pass: if validation is still too small or has too few positives,
    # add more cities greedily, without emptying any training region.
    while True:
        val_n, val_pos = selected_stats(selected)

        enough_patches = val_n >= target_val_patches
        enough_positives = val_pos >= min_val_positive_patches

        if enough_patches and enough_positives:
            break

        candidates = city_stats[~city_stats["city"].isin(selected)].copy()

        if candidates.empty:
            warnings.append("No more candidate cities are available for validation.")
            break

        candidates = candidates[candidates["city"].apply(region_would_keep_train_city)].copy()

        if candidates.empty:
            warnings.append(
                "No more candidate cities can be added to validation without removing a region from training."
            )
            break

        candidates["new_val_patches"] = candidates["patches"] + val_n
        candidates["new_val_positive"] = candidates["positive_patches"] + val_pos
        candidates["score"] = (
            (candidates["new_val_patches"] - target_val_patches).abs()
            - 0.10 * candidates["new_val_positive"].clip(upper=max(1, min_val_positive_patches))
        )

        chosen = candidates.sort_values(["score", "patches", "city"]).iloc[0]["city"]
        selected.append(str(chosen))

    if not selected:
        fail("Could not select any validation city.")

    selected = sorted(set(selected))
    val_n, val_pos = selected_stats(selected)

    if val_n < min_val_patches:
        warnings.append(
            f"Validation set has only {val_n:,} patches, below requested minimum {min_val_patches:,}."
        )

    if val_pos < min_val_positive_patches:
        warnings.append(
            f"Validation set has only {val_pos:,} positive patches, "
            f"below requested minimum {min_val_positive_patches:,}."
        )

    return selected, city_stats, warnings


def validate_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    city_col: str,
    region_col: str,
    patch_id_col: str,
    heldout_region: str,
    min_positive_patches: int,
) -> List[str]:
    warnings: List[str] = []

    train_patch_ids = set(train_df[patch_id_col].astype(str))
    val_patch_ids = set(val_df[patch_id_col].astype(str))
    test_patch_ids = set(test_df[patch_id_col].astype(str))

    if train_patch_ids & val_patch_ids:
        warnings.append("Patch leakage detected between train and validation.")

    if train_patch_ids & test_patch_ids:
        warnings.append("Patch leakage detected between train and test.")

    if val_patch_ids & test_patch_ids:
        warnings.append("Patch leakage detected between validation and test.")

    train_cities = set(train_df[city_col].astype(str))
    val_cities = set(val_df[city_col].astype(str))
    test_cities = set(test_df[city_col].astype(str))

    if train_cities & val_cities:
        warnings.append("City leakage detected between train and validation.")

    if train_cities & test_cities:
        warnings.append("City overlap detected between train and test.")

    if val_cities & test_cities:
        warnings.append("City overlap detected between validation and test.")

    test_regions = set(test_df[region_col].astype(str))
    if test_regions != {heldout_region}:
        warnings.append(
            f"Test set should contain only held-out region '{heldout_region}', "
            f"but found {sorted(test_regions)}."
        )

    train_regions = set(train_df[region_col].astype(str))
    val_regions = set(val_df[region_col].astype(str))

    if heldout_region in train_regions:
        warnings.append(f"Held-out region '{heldout_region}' appears in training set.")

    if heldout_region in val_regions:
        warnings.append(f"Held-out region '{heldout_region}' appears in validation set.")

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        if split_df.empty:
            warnings.append(f"{split_name} split is empty.")
            continue

        positive_patches = int(split_df["__positive_patch"].sum())
        empty_patches = int(len(split_df) - positive_patches)

        if positive_patches == 0:
            warnings.append(f"{split_name} split has no positive patches.")

        if empty_patches == 0:
            warnings.append(f"{split_name} split has no empty patches.")

        if positive_patches < min_positive_patches:
            warnings.append(
                f"{split_name} split has only {positive_patches:,} positive patches "
                f"(< {min_positive_patches:,})."
            )

    return warnings


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if math.isnan(value):
            return None
        return value

    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        return obj

    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple, set)) else False:
        return None

    return obj


def format_pct(value: Any) -> str:
    try:
        value = float(value)
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    except Exception:
        return "NA"


def write_markdown_summary(
    path: Path,
    summary_df: pd.DataFrame,
    fold_details: Dict[str, Any],
    manifest_path: Path,
    target_modality: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
) -> None:
    lines: List[str] = []

    lines.append("# UPerNet CROMA Leave-One-Region-Out Split Summary")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- Target modality: `{target_modality}`")
    lines.append(f"- Patch size: `{patch_size}`")
    lines.append(f"- Stride: `{stride}`")
    lines.append(f"- Edge mode: `{edge_mode}`")
    lines.append("")
    lines.append("## Fold Summary")
    lines.append("")
    lines.append(
        "| Held-out region | Train patches | Val patches | Test patches | "
        "Train pos | Val pos | Test pos | Train favela % | Val favela % | Test favela % | Warnings |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )

    for _, row in summary_df.iterrows():
        warnings_text = str(row.get("warnings", ""))
        if warnings_text.strip() == "":
            warnings_text = "None"

        lines.append(
            f"| {row['heldout_region']} "
            f"| {int(row['train_patches']):,} "
            f"| {int(row['val_patches']):,} "
            f"| {int(row['test_patches']):,} "
            f"| {int(row['train_positive_patches']):,} "
            f"| {int(row['val_positive_patches']):,} "
            f"| {int(row['test_positive_patches']):,} "
            f"| {format_pct(row['train_favela_pixel_pct'])} "
            f"| {format_pct(row['val_favela_pixel_pct'])} "
            f"| {format_pct(row['test_favela_pixel_pct'])} "
            f"| {warnings_text} |"
        )

    lines.append("")
    lines.append("## Fold Details")
    lines.append("")

    for heldout_region, details in fold_details.items():
        lines.append(f"### Held-out region: {heldout_region}")
        lines.append("")

        for split_name in ["train", "val", "test"]:
            stats = details[split_name]
            lines.append(f"#### {split_name}")
            lines.append("")
            lines.append(f"- Patches: {stats['patches']:,}")
            lines.append(f"- Positive patches: {stats['positive_patches']:,}")
            lines.append(f"- Empty patches: {stats['empty_patches']:,}")
            lines.append(f"- Positive patch percentage: {stats['positive_patch_pct']:.3f}%")
            lines.append(f"- Favela pixel percentage: {format_pct(stats['favela_pixel_pct'])}%")
            lines.append(f"- Regions: {', '.join(stats['regions'])}")
            lines.append(f"- Cities: {', '.join(stats['cities'])}")
            lines.append("")

        warnings_list = details.get("warnings", [])
        lines.append("#### Warnings")
        lines.append("")
        if warnings_list:
            for warning in warnings_list:
                lines.append(f"- {warning}")
        else:
            lines.append("- None")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_loro_splits(args: argparse.Namespace) -> None:
    instance_root = Path(args.instance_root)
    manifest_path = Path(args.manifest) if args.manifest else default_manifest_path(
        instance_root=instance_root,
        patch_size=args.patch_size,
        stride=args.stride,
        edge_mode=args.edge_mode,
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
        instance_root=instance_root,
        patch_size=args.patch_size,
        stride=args.stride,
        edge_mode=args.edge_mode,
    )

    log("=" * 100)
    log("UPerNet/CROMA Leave-One-Region-Out Split Manifest Builder")
    log("=" * 100)
    log(f"Instance root: {instance_root}")
    log(f"Manifest:      {manifest_path}")
    log(f"Output dir:    {output_dir}")
    log(f"Target modality: {args.target_modality}")
    log(f"Patch size: {args.patch_size}")
    log(f"Stride: {args.stride}")
    log(f"Edge mode: {args.edge_mode}")
    log("=" * 100)

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    df = read_manifest(manifest_path)

    city_col = find_column(
        df,
        CITY_COLUMN_CANDIDATES,
        explicit=args.city_col,
        required=True,
        purpose="city column",
    )
    region_col = find_column(
        df,
        REGION_COLUMN_CANDIDATES,
        explicit=args.region_col,
        required=True,
        purpose="region column",
    )
    modality_col = find_column(
        df,
        MODALITY_COLUMN_CANDIDATES,
        explicit=args.modality_col,
        required=False,
        purpose="modality column",
    )

    log("")
    log("Detected columns:")
    log(f"  city column:     {city_col}")
    log(f"  region column:   {region_col}")
    log(f"  modality column: {modality_col if modality_col else 'NOT FOUND'}")
    log("")

    df = filter_target_modality(
        df=df,
        modality_col=modality_col,
        target_modality=args.target_modality,
    )

    df, patch_id_col = build_patch_id(
        df=df,
        city_col=city_col,
        region_col=region_col,
        explicit_patch_id_col=args.patch_id_col,
    )

    df, positive_source, pos_pixel_col, total_pixel_col, favela_pct_col = infer_positive_column(
        df=df,
        explicit_positive_col=args.positive_col,
        positive_threshold=args.positive_threshold,
    )

    log("")
    log("Detected label/statistics columns:")
    log(f"  positive source:       {positive_source}")
    log(f"  positive pixels col:   {pos_pixel_col if pos_pixel_col else 'NOT FOUND'}")
    log(f"  total pixels col:      {total_pixel_col if total_pixel_col else 'NOT FOUND'}")
    log(f"  favela percentage col: {favela_pct_col if favela_pct_col else 'NOT FOUND'}")
    log("")

    unique_df = deduplicate_patches(df, patch_id_col=patch_id_col)

    unique_regions = sorted(unique_df[region_col].dropna().astype(str).unique().tolist())

    log("")
    log("[5/8] Regions detected after filtering and deduplication:")
    for region in unique_regions:
        n_region = int((unique_df[region_col].astype(str) == region).sum())
        log(f"      {region}: {n_region:,} patches")

    missing_expected = [r for r in EXPECTED_REGIONS if r not in unique_regions]
    if missing_expected:
        warn(f"Expected regions not found in manifest: {missing_expected}")

    extra_regions = [r for r in unique_regions if r not in EXPECTED_REGIONS]
    if extra_regions:
        warn(f"Additional regions found outside expected list: {extra_regions}")

    fold_regions = [r for r in EXPECTED_REGIONS if r in unique_regions]
    if not fold_regions:
        fail("None of the expected LORO regions were found in the manifest.")

    summary_rows: List[Dict[str, Any]] = []
    fold_details: Dict[str, Any] = {}

    log("")
    log("[6/8] Building LORO folds")
    log("")

    for heldout_region in fold_regions:
        log("-" * 100)
        log(f"Building fold with held-out region: {heldout_region}")

        test_df = unique_df[unique_df[region_col].astype(str) == heldout_region].copy()
        trainval_df = unique_df[unique_df[region_col].astype(str) != heldout_region].copy()

        if test_df.empty:
            fail(f"Test set is empty for held-out region: {heldout_region}")

        if trainval_df.empty:
            fail(f"Train/validation pool is empty for held-out region: {heldout_region}")

        val_cities, city_stats, selection_warnings = choose_validation_cities(
            trainval_df=trainval_df,
            city_col=city_col,
            region_col=region_col,
            pos_pixel_col=pos_pixel_col,
            total_pixel_col=total_pixel_col,
            favela_pct_col=favela_pct_col,
            val_fraction=args.val_fraction,
            min_val_patches=args.min_val_patches,
            min_val_positive_patches=args.min_val_positive_patches,
        )

        val_df = trainval_df[trainval_df[city_col].astype(str).isin(val_cities)].copy()
        train_df = trainval_df[~trainval_df[city_col].astype(str).isin(val_cities)].copy()

        train_df["upernet_split"] = "train"
        val_df["upernet_split"] = "val"
        test_df["upernet_split"] = "test"
        train_df["loro_heldout_region"] = heldout_region
        val_df["loro_heldout_region"] = heldout_region
        test_df["loro_heldout_region"] = heldout_region

        fold_warnings = []
        fold_warnings.extend(selection_warnings)
        fold_warnings.extend(
            validate_fold(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                city_col=city_col,
                region_col=region_col,
                patch_id_col=patch_id_col,
                heldout_region=heldout_region,
                min_positive_patches=args.min_split_positive_patches,
            )
        )

        train_stats = split_stats(
            train_df,
            city_col=city_col,
            region_col=region_col,
            patch_id_col=patch_id_col,
            pos_pixel_col=pos_pixel_col,
            total_pixel_col=total_pixel_col,
            favela_pct_col=favela_pct_col,
        )
        val_stats = split_stats(
            val_df,
            city_col=city_col,
            region_col=region_col,
            patch_id_col=patch_id_col,
            pos_pixel_col=pos_pixel_col,
            total_pixel_col=total_pixel_col,
            favela_pct_col=favela_pct_col,
        )
        test_stats = split_stats(
            test_df,
            city_col=city_col,
            region_col=region_col,
            patch_id_col=patch_id_col,
            pos_pixel_col=pos_pixel_col,
            total_pixel_col=total_pixel_col,
            favela_pct_col=favela_pct_col,
        )

        region_slug = slug_region(heldout_region)

        train_path = output_dir / f"loro_fold_{region_slug}_train.csv"
        val_path = output_dir / f"loro_fold_{region_slug}_val.csv"
        test_path = output_dir / f"loro_fold_{region_slug}_test.csv"
        city_stats_path = output_dir / f"loro_fold_{region_slug}_city_candidate_stats.csv"

        if not args.dry_run:
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            city_stats.to_csv(city_stats_path, index=False)

        log(f"  Train patches: {train_stats['patches']:,}")
        log(f"  Val patches:   {val_stats['patches']:,}")
        log(f"  Test patches:  {test_stats['patches']:,}")
        log(f"  Val cities:    {', '.join(val_stats['cities'])}")

        if fold_warnings:
            log("  Warnings:")
            for warning in fold_warnings:
                warn(f"{heldout_region}: {warning}")
        else:
            log("  Warnings: none")

        summary_rows.append(
            {
                "heldout_region": heldout_region,

                "train_patches": train_stats["patches"],
                "val_patches": val_stats["patches"],
                "test_patches": test_stats["patches"],

                "train_positive_patches": train_stats["positive_patches"],
                "val_positive_patches": val_stats["positive_patches"],
                "test_positive_patches": test_stats["positive_patches"],

                "train_empty_patches": train_stats["empty_patches"],
                "val_empty_patches": val_stats["empty_patches"],
                "test_empty_patches": test_stats["empty_patches"],

                "train_positive_patch_pct": train_stats["positive_patch_pct"],
                "val_positive_patch_pct": val_stats["positive_patch_pct"],
                "test_positive_patch_pct": test_stats["positive_patch_pct"],

                "train_favela_pixel_pct": train_stats["favela_pixel_pct"],
                "val_favela_pixel_pct": val_stats["favela_pixel_pct"],
                "test_favela_pixel_pct": test_stats["favela_pixel_pct"],

                "train_n_cities": train_stats["n_cities"],
                "val_n_cities": val_stats["n_cities"],
                "test_n_cities": test_stats["n_cities"],

                "train_cities": ";".join(train_stats["cities"]),
                "val_cities": ";".join(val_stats["cities"]),
                "test_cities": ";".join(test_stats["cities"]),

                "train_regions": ";".join(train_stats["regions"]),
                "val_regions": ";".join(val_stats["regions"]),
                "test_regions": ";".join(test_stats["regions"]),

                "warnings": " | ".join(fold_warnings),
            }
        )

        fold_details[heldout_region] = {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
            "warnings": fold_warnings,
            "paths": {
                "train": str(train_path),
                "val": str(val_path),
                "test": str(test_path),
                "city_candidate_stats": str(city_stats_path),
            },
        }

    log("")
    log("[7/8] Writing summary files")

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = output_dir / f"loro_split_summary_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    summary_json = output_dir / f"loro_split_summary_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.json"
    summary_md = output_dir / f"loro_split_summary_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.md"

    metadata = {
        "instance_root": str(instance_root),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "target_modality": args.target_modality,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "edge_mode": args.edge_mode,
        "city_column": city_col,
        "region_column": region_col,
        "modality_column": modality_col,
        "patch_id_column": patch_id_col,
        "positive_source": positive_source,
        "positive_pixel_column": pos_pixel_col,
        "total_pixel_column": total_pixel_col,
        "favela_percentage_column": favela_pct_col,
        "n_unique_patches": int(len(unique_df)),
        "regions": unique_regions,
        "folds": fold_details,
    }

    if not args.dry_run:
        summary_df.to_csv(summary_csv, index=False)

        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(metadata), f, indent=2, ensure_ascii=False)

        write_markdown_summary(
            path=summary_md,
            summary_df=summary_df,
            fold_details=fold_details,
            manifest_path=manifest_path,
            target_modality=args.target_modality,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
        )

    log(f"      Summary CSV:  {summary_csv}")
    log(f"      Summary JSON: {summary_json}")
    log(f"      Summary MD:   {summary_md}")

    log("")
    log("[8/8] Done")
    log("=" * 100)

    if args.dry_run:
        log("Dry run enabled: no files were written.")
    else:
        log(f"Split manifests written to:\n{output_dir}")

    total_warnings = int(summary_df["warnings"].astype(str).str.len().gt(0).sum())
    if total_warnings > 0:
        warn(f"{total_warnings} folds contain warnings. Inspect the Markdown summary carefully.")
    else:
        log("No fold-level warnings were reported.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Leave-One-Region-Out split manifests for UPerNet/CROMA segmentation."
    )

    parser.add_argument(
        "--instance-root",
        required=True,
        help="Dataset instance root, e.g. D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional explicit path to the CROMA comparison manifest. If omitted, the default path is inferred.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. If omitted, the default metadata/upernet_croma directory is used.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size used in the manifest.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=112,
        help="Patch stride used in the manifest.",
    )
    parser.add_argument(
        "--edge-mode",
        default="cover",
        help="Edge mode used in the manifest name.",
    )
    parser.add_argument(
        "--target-modality",
        default="s2_s1_snap_vv_vh",
        help="Target modality to keep from the multi-modality CROMA manifest.",
    )

    parser.add_argument(
        "--city-col",
        default=None,
        help="Optional explicit city column name.",
    )
    parser.add_argument(
        "--region-col",
        default=None,
        help="Optional explicit region column name.",
    )
    parser.add_argument(
        "--modality-col",
        default=None,
        help="Optional explicit modality column name.",
    )
    parser.add_argument(
        "--patch-id-col",
        default=None,
        help="Optional explicit unique patch id column name.",
    )
    parser.add_argument(
        "--positive-col",
        default=None,
        help="Optional explicit column used to decide whether a patch is positive.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help="Threshold above which a patch is considered positive.",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.20,
        help="Approximate validation fraction from the non-test regions.",
    )
    parser.add_argument(
        "--min-val-patches",
        type=int,
        default=100,
        help="Minimum desired number of validation patches.",
    )
    parser.add_argument(
        "--min-val-positive-patches",
        type=int,
        default=20,
        help="Minimum desired number of positive validation patches.",
    )
    parser.add_argument(
        "--min-split-positive-patches",
        type=int,
        default=5,
        help="Warning threshold for too few positive patches in any split.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks without writing files.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    build_loro_splits(parse_args())