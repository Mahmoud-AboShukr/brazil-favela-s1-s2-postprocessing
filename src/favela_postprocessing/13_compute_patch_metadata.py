#!/usr/bin/env python3
"""
Compute patch-level metadata for the Brazil favela segmentation dataset.

Purpose
-------
This script reads the patch tiling index produced by:

    12_build_patch_tiling_index.py

and computes patch-level quality/label metadata by reading the corresponding raster windows.

It does NOT export image arrays and does NOT create H5 files.

For each patch it computes:
    - label_positive_pixels
    - label_positive_percent
    - is_positive_patch
    - S2 nodata/all-zero percentage
    - S1 nodata percentage
    - optional cloud percentage if a cloud mask is available
    - basic quality flags useful for later filtering/sampling

Inputs
------
Default tiling index:

    <output_root>/metadata/patch_tiling_index_train_covered_region_test_ps512_st512_cover.csv

Outputs
-------
CSV:
    <output_root>/metadata/patch_metadata_train_covered_region_test_ps512_st512_cover.csv

Parquet, if pyarrow/fastparquet is available:
    <output_root>/metadata/patch_metadata_train_covered_region_test_ps512_st512_cover.parquet

GeoPackage, if geopandas/shapely are available:
    <output_root>/metadata/patch_metadata_train_covered_region_test_ps512_st512_cover.gpkg

Summary:
    <output_root>/qc/patch_metadata_summary_train_covered_region_test_ps512_st512_cover.csv
    <output_root>/metadata/patch_metadata_train_covered_region_test_ps512_st512_cover.md

Example
-------
    python3 src/favela_postprocessing/13_compute_patch_metadata.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import yaml
from tqdm import tqdm


SCRIPT_NAME = "13_compute_patch_metadata.py"

DEFAULT_SPLIT_STRATEGY = "train_covered_region_test"
DEFAULT_PATCH_SIZE = 512
DEFAULT_STRIDE = 512
DEFAULT_EDGE_MODE = "cover"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute patch-level metadata from a patch tiling index."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--tiling-index",
        type=Path,
        default=None,
        help=(
            "Explicit patch tiling index CSV. If omitted, uses the default "
            "train_covered_region_test ps512 st512 cover file."
        ),
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default=DEFAULT_SPLIT_STRATEGY,
        help=f"Split strategy used for default tiling-index lookup. Default: {DEFAULT_SPLIT_STRATEGY}",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Patch size used for default tiling-index lookup. Default: {DEFAULT_PATCH_SIZE}",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Stride used for default tiling-index lookup. Default: {DEFAULT_STRIDE}",
    )
    parser.add_argument(
        "--edge-mode",
        type=str,
        default=DEFAULT_EDGE_MODE,
        help=f"Edge mode used for default tiling-index lookup. Default: {DEFAULT_EDGE_MODE}",
    )
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="Compute metadata for only one city. Can be repeated.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )
    parser.add_argument(
        "--no-gpkg",
        action="store_true",
        help="Do not write GeoPackage output.",
    )
    parser.add_argument(
        "--treat-all-zero-s2-as-nodata",
        action="store_true",
        default=True,
        help="Treat S2 pixels where all bands are zero as nodata. Default: true.",
    )
    parser.add_argument(
        "--do-not-treat-all-zero-s2-as-nodata",
        dest="treat_all_zero_s2_as_nodata",
        action="store_false",
        help="Disable treating all-zero S2 pixels as nodata.",
    )
    parser.add_argument(
        "--nodata-threshold-percent",
        type=float,
        default=10.0,
        help="Threshold for quality flag passes_nodata_filter. Default: 10 percent.",
    )
    parser.add_argument(
        "--cloud-threshold-percent",
        type=float,
        default=20.0,
        help="Threshold for quality flag passes_cloud_filter. Default: 20 percent.",
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


def default_tiling_index_path(
    output_root: Path,
    split_strategy: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
) -> Path:
    suffix = f"{split_strategy}_ps{patch_size}_st{stride}_{edge_mode}"
    return output_root / "metadata" / f"patch_tiling_index_{suffix}.csv"


def suffix_from_tiling_index(path: Path) -> str:
    name = path.stem

    prefix = "patch_tiling_index_"

    if name.startswith(prefix):
        return name[len(prefix):]

    return name


def load_tiling_index(
    path: Path,
    selected_cities: Optional[Sequence[str]],
    max_patches: Optional[int],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Patch tiling index not found: {path}\n"
            f"Run 12_build_patch_tiling_index.py first."
        )

    df = pd.read_csv(path)

    required = [
        "patch_id",
        "city",
        "split_strategy",
        "fold_id",
        "split",
        "source_s2_path",
        "source_s1_path",
        "source_label_path",
        "row_start",
        "row_end",
        "col_start",
        "col_end",
        "patch_height",
        "patch_width",
        "bbox_minlon",
        "bbox_minlat",
        "bbox_maxlon",
        "bbox_maxlat",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Tiling index is missing required columns: {missing}")

    df = df.copy()
    df["city"] = df["city"].map(normalize_city_name)

    if selected_cities:
        selected = {normalize_city_name(city) for city in selected_cities}
        df = df[df["city"].isin(selected)].copy()

    if max_patches is not None:
        df = df.head(max_patches).copy()

    if df.empty:
        raise RuntimeError("No patch rows selected.")

    return df.reset_index(drop=True)


def window_from_row(row: pd.Series) -> Window:
    row_start = int(row["row_start"])
    col_start = int(row["col_start"])
    patch_height = int(row["patch_height"])
    patch_width = int(row["patch_width"])

    return Window(
        col_off=col_start,
        row_off=row_start,
        width=patch_width,
        height=patch_height,
    )


def valid_mask_for_array(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """
    Return per-pixel valid mask for an array.

    If arr is 2D:
        returns [H, W]

    If arr is 3D [C, H, W]:
        returns [H, W], valid only where all channels are valid.
    """
    if arr.ndim == 2:
        valid = np.isfinite(arr)
        if nodata is not None and np.isfinite(nodata):
            valid &= arr != nodata
        return valid

    if arr.ndim == 3:
        valid = np.all(np.isfinite(arr), axis=0)
        if nodata is not None and np.isfinite(nodata):
            valid &= np.all(arr != nodata, axis=0)
        return valid

    raise ValueError(f"Unsupported array ndim: {arr.ndim}")


def all_zero_mask(arr: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    if arr.ndim == 2:
        return np.abs(arr) <= tolerance

    if arr.ndim == 3:
        return np.all(np.abs(arr) <= tolerance, axis=0)

    raise ValueError(f"Unsupported array ndim: {arr.ndim}")


def count_percent(count: int, total: int) -> float:
    if total <= 0:
        return math.nan
    return 100.0 * count / total


def safe_min(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.min(values))


def safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.max(values))


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.mean(values))


def find_cloud_mask(output_root: Path, city: str) -> Optional[Path]:
    """
    Try to find an auxiliary cloud mask for a city.

    This is intentionally flexible because previous S2 pipelines may have produced
    slightly different filenames.
    """
    root = output_root / "auxiliary" / "cloud_masks"

    candidates = [
        root / city / f"{city}_cloud_mask.tif",
        root / city / f"{city}_cloud_mask_10m.tif",
        root / city / f"{city}_s2_cloud_mask.tif",
        root / city / f"{city}_s2_cloud_mask_10m.tif",
        root / f"{city}_cloud_mask.tif",
        root / f"{city}_cloud_mask_10m.tif",
        root / f"{city}_s2_cloud_mask.tif",
        root / f"{city}_s2_cloud_mask_10m.tif",
    ]

    for path in candidates:
        if path.exists():
            return path

    city_dir = root / city

    if city_dir.exists():
        tif_files = sorted(city_dir.glob("*.tif"))

        if len(tif_files) == 1:
            return tif_files[0]

    return None


def compute_label_metadata(label_path: Path, window: Window) -> Dict[str, Any]:
    """
    Compute label metadata for a binary favela mask.

    Important:
    ----------
    For our final labels:
        0 = background
        1 = favela

    Therefore, value 0 must be treated as a valid class, not as nodata.
    Some GeoTIFF readers may report nodata=0 for binary masks, but applying that
    literally would incorrectly mark all background pixels as nodata.

    We therefore define label-valid pixels as finite pixels whose values are 0 or 1.
    Any finite value outside {0, 1} is reported as non-binary.
    """
    with rasterio.open(label_path) as src:
        arr = src.read(1, window=window)

        total_pixels = int(arr.size)

        finite = np.isfinite(arr)

        binary_valid = finite & ((arr == 0) | (arr == 1))
        valid_pixels = int(binary_valid.sum())
        nodata_pixels = total_pixels - valid_pixels

        positive = binary_valid & (arr == 1)
        positive_pixels = int(positive.sum())

        background = binary_valid & (arr == 0)
        background_pixels = int(background.sum())

        non_binary = finite & ~((arr == 0) | (arr == 1))
        non_binary_pixels = int(non_binary.sum())

        return {
            "label_total_pixels": total_pixels,
            "label_valid_pixels": valid_pixels,
            "label_nodata_pixels": nodata_pixels,
            "label_nodata_percent": count_percent(nodata_pixels, total_pixels),
            "label_positive_pixels": positive_pixels,
            "label_background_pixels": background_pixels,
            "label_non_binary_pixels": non_binary_pixels,
            "label_positive_percent": count_percent(positive_pixels, total_pixels),
            "label_positive_percent_valid": count_percent(positive_pixels, valid_pixels),
            "is_positive_patch": bool(positive_pixels > 0),
            "label_min": safe_min(arr[binary_valid]),
            "label_max": safe_max(arr[binary_valid]),
            "label_mean": safe_mean(arr[binary_valid]),
            "label_reported_nodata": src.nodata,
        }

def compute_s2_metadata(
    s2_path: Path,
    window: Window,
    treat_all_zero_as_nodata: bool,
) -> Dict[str, Any]:
    with rasterio.open(s2_path) as src:
        arr = src.read(window=window).astype("float32", copy=False)

        total_pixels = int(arr.shape[1] * arr.shape[2])

        valid = valid_mask_for_array(arr, src.nodata)

        all_zero = all_zero_mask(arr)

        if treat_all_zero_as_nodata:
            valid = valid & ~all_zero

        valid_pixels = int(valid.sum())
        nodata_pixels = total_pixels - valid_pixels
        all_zero_pixels = int(all_zero.sum())

        return {
            "s2_band_count": int(src.count),
            "s2_total_pixels": total_pixels,
            "s2_valid_pixels": valid_pixels,
            "s2_nodata_pixels": nodata_pixels,
            "s2_nodata_percent": count_percent(nodata_pixels, total_pixels),
            "s2_all_zero_pixels": all_zero_pixels,
            "s2_all_zero_percent": count_percent(all_zero_pixels, total_pixels),
            "s2_treat_all_zero_as_nodata": bool(treat_all_zero_as_nodata),
        }


def compute_s1_metadata(s1_path: Path, window: Window) -> Dict[str, Any]:
    with rasterio.open(s1_path) as src:
        arr = src.read(window=window).astype("float32", copy=False)

        total_pixels = int(arr.shape[1] * arr.shape[2])

        valid = valid_mask_for_array(arr, src.nodata)

        valid_pixels = int(valid.sum())
        nodata_pixels = total_pixels - valid_pixels

        return {
            "s1_band_count": int(src.count),
            "s1_total_pixels": total_pixels,
            "s1_valid_pixels": valid_pixels,
            "s1_nodata_pixels": nodata_pixels,
            "s1_nodata_percent": count_percent(nodata_pixels, total_pixels),
        }


def compute_cloud_metadata(
    output_root: Path,
    city: str,
    window: Window,
) -> Dict[str, Any]:
    cloud_path = find_cloud_mask(output_root, city)

    if cloud_path is None:
        return {
            "has_cloud_mask": False,
            "cloud_mask_path": "",
            "cloud_total_pixels": math.nan,
            "cloud_valid_pixels": math.nan,
            "cloud_pixels": math.nan,
            "cloud_percent": math.nan,
        }

    with rasterio.open(cloud_path) as src:
        arr = src.read(1, window=window)

        total_pixels = int(arr.size)
        valid = valid_mask_for_array(arr, src.nodata)

        valid_pixels = int(valid.sum())

        cloud = valid & (arr > 0)
        cloud_pixels = int(cloud.sum())

        return {
            "has_cloud_mask": True,
            "cloud_mask_path": str(cloud_path),
            "cloud_total_pixels": total_pixels,
            "cloud_valid_pixels": valid_pixels,
            "cloud_pixels": cloud_pixels,
            "cloud_percent": count_percent(cloud_pixels, total_pixels),
        }


def compute_patch_metadata(
    output_root: Path,
    row: pd.Series,
    treat_all_zero_s2_as_nodata: bool,
    nodata_threshold_percent: float,
    cloud_threshold_percent: float,
) -> Dict[str, Any]:
    city = normalize_city_name(str(row["city"]))

    s2_path = Path(str(row["source_s2_path"]))
    s1_path = Path(str(row["source_s1_path"]))
    label_path = Path(str(row["source_label_path"]))

    if not s2_path.exists():
        raise FileNotFoundError(f"S2 path does not exist for patch {row['patch_id']}: {s2_path}")

    if not s1_path.exists():
        raise FileNotFoundError(f"S1 path does not exist for patch {row['patch_id']}: {s1_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label path does not exist for patch {row['patch_id']}: {label_path}")

    window = window_from_row(row)

    label_meta = compute_label_metadata(label_path, window)
    s2_meta = compute_s2_metadata(
        s2_path=s2_path,
        window=window,
        treat_all_zero_as_nodata=treat_all_zero_s2_as_nodata,
    )
    s1_meta = compute_s1_metadata(s1_path, window)
    cloud_meta = compute_cloud_metadata(output_root, city, window)

    max_nodata_percent = max(
        float(s2_meta["s2_nodata_percent"]),
        float(s1_meta["s1_nodata_percent"]),
        float(label_meta["label_nodata_percent"]),
    )

    has_cloud_mask = bool(cloud_meta["has_cloud_mask"])

    if has_cloud_mask:
        cloud_percent = float(cloud_meta["cloud_percent"])
        passes_cloud_filter = cloud_percent <= cloud_threshold_percent
    else:
        cloud_percent = math.nan
        passes_cloud_filter = True

    passes_nodata_filter = max_nodata_percent <= nodata_threshold_percent

    label_positive_percent = float(label_meta["label_positive_percent"])

    return {
        **label_meta,
        **s2_meta,
        **s1_meta,
        **cloud_meta,
        "max_nodata_percent": max_nodata_percent,
        "passes_nodata_filter": bool(passes_nodata_filter),
        "passes_cloud_filter": bool(passes_cloud_filter),
        "passes_basic_quality_filter": bool(passes_nodata_filter and passes_cloud_filter),
        "positive_gt_0pct": bool(label_positive_percent > 0.0),
        "positive_gt_0_5pct": bool(label_positive_percent > 0.5),
        "positive_gt_1pct": bool(label_positive_percent > 1.0),
        "positive_gt_2pct": bool(label_positive_percent > 2.0),
        "recommended_sampling_group": "positive" if label_positive_percent > 0.0 else "negative",
    }


def summarize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "split_strategy",
        "fold_id",
        "heldout_region",
        "split",
        "region",
    ]

    summary = (
        df.groupby(group_cols)
        .agg(
            patch_count=("patch_id", "count"),
            city_count=("city", "nunique"),
            positive_patch_count=("is_positive_patch", "sum"),
            mean_label_positive_percent=("label_positive_percent", "mean"),
            max_label_positive_percent=("label_positive_percent", "max"),
            mean_s2_nodata_percent=("s2_nodata_percent", "mean"),
            mean_s1_nodata_percent=("s1_nodata_percent", "mean"),
            mean_max_nodata_percent=("max_nodata_percent", "mean"),
            quality_pass_count=("passes_basic_quality_filter", "sum"),
            cloud_mask_available_count=("has_cloud_mask", "sum"),
        )
        .reset_index()
        .sort_values(["split_strategy", "fold_id", "split", "region"])
    )

    summary["positive_patch_percent"] = (
        100.0 * summary["positive_patch_count"] / summary["patch_count"]
    )

    summary["quality_pass_percent"] = (
        100.0 * summary["quality_pass_count"] / summary["patch_count"]
    )

    return summary


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
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    md_path: Path,
    suffix: str,
    nodata_threshold_percent: float,
    cloud_threshold_percent: float,
) -> None:
    ensure_dir(md_path.parent)

    total_patches = len(df)
    positive_patches = int(df["is_positive_patch"].sum())
    quality_pass = int(df["passes_basic_quality_filter"].sum())

    lines: List[str] = []

    lines.append("# Patch Metadata Summary")
    lines.append("")
    lines.append(f"Generated by `{SCRIPT_NAME}`.")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "This file summarizes patch-level metadata computed from the patch tiling index. "
        "It is used for filtering, class-imbalance control, sampling strategy design, "
        "and later ML dataloader construction."
    )
    lines.append("")

    lines.append("## Source")
    lines.append("")
    lines.append(f"- Tiling suffix: `{suffix}`")
    lines.append("")

    lines.append("## Main counts")
    lines.append("")

    counts_df = pd.DataFrame(
        [
            {"metric": "Total patches", "value": total_patches},
            {"metric": "Positive patches", "value": positive_patches},
            {"metric": "Negative patches", "value": total_patches - positive_patches},
            {"metric": "Positive patch percent", "value": f"{100.0 * positive_patches / total_patches:.2f}%"},
            {"metric": "Quality-pass patches", "value": quality_pass},
            {"metric": "Quality-pass percent", "value": f"{100.0 * quality_pass / total_patches:.2f}%"},
            {"metric": "Nodata threshold", "value": f"{nodata_threshold_percent}%"},
            {"metric": "Cloud threshold", "value": f"{cloud_threshold_percent}%"},
        ]
    )

    lines.append(df_to_markdown_table(counts_df))
    lines.append("")

    lines.append("## Summary by split and region")
    lines.append("")

    display_cols = [
        "split_strategy",
        "fold_id",
        "split",
        "region",
        "patch_count",
        "positive_patch_count",
        "positive_patch_percent",
        "mean_label_positive_percent",
        "mean_s2_nodata_percent",
        "mean_s1_nodata_percent",
        "quality_pass_percent",
    ]

    display = summary_df[display_cols].copy()

    for col in [
        "positive_patch_percent",
        "mean_label_positive_percent",
        "mean_s2_nodata_percent",
        "mean_s1_nodata_percent",
        "quality_pass_percent",
    ]:
        display[col] = display[col].map(lambda x: f"{float(x):.3f}")

    lines.append(df_to_markdown_table(display))
    lines.append("")

    lines.append("## Patch filtering interpretation")
    lines.append("")
    lines.append("- `is_positive_patch=True` means the patch contains at least one positive favela-label pixel.")
    lines.append("- `label_positive_percent` is the percentage of pixels in the patch labelled as favela.")
    lines.append("- `passes_nodata_filter=True` means S2, S1, and label nodata percentages are below the chosen nodata threshold.")
    lines.append("- `passes_cloud_filter=True` means cloud percentage is below threshold, or no cloud mask was available.")
    lines.append("- `passes_basic_quality_filter=True` means both nodata and cloud filters passed.")
    lines.append("")

    lines.append("## Recommended next step")
    lines.append("")
    lines.append(
        "Use this metadata to define patch filtering and sampling rules, for example: "
        "all positive patches plus a controlled ratio of negative patches."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_parquet(df: pd.DataFrame, path: Path) -> bool:
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception as exc:
        print(f"[WARN] Could not write Parquet. Reason: {exc}")
        return False


def write_gpkg(df: pd.DataFrame, path: Path) -> bool:
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except Exception as exc:
        print(f"[WARN] Could not import geopandas/shapely. Skipping GeoPackage. Reason: {exc}")
        return False

    ensure_dir(path.parent)

    geometry = [
        box(row.bbox_minlon, row.bbox_minlat, row.bbox_maxlon, row.bbox_maxlat)
        for row in df.itertuples(index=False)
    ]

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=geometry,
        crs="EPSG:4326",
    )

    gdf.to_file(path, driver="GPKG")
    return True


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))
    metadata_root = output_root / "metadata"
    qc_root = output_root / "qc"

    ensure_dir(metadata_root)
    ensure_dir(qc_root)

    if args.tiling_index is None:
        tiling_index_path = default_tiling_index_path(
            output_root=output_root,
            split_strategy=args.split_strategy,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
        )
    else:
        tiling_index_path = args.tiling_index

    suffix = suffix_from_tiling_index(tiling_index_path)

    csv_path = metadata_root / f"patch_metadata_{suffix}.csv"
    parquet_path = metadata_root / f"patch_metadata_{suffix}.parquet"
    gpkg_path = metadata_root / f"patch_metadata_{suffix}.gpkg"
    md_path = metadata_root / f"patch_metadata_{suffix}.md"
    summary_path = qc_root / f"patch_metadata_summary_{suffix}.csv"

    print("[INFO] Compute patch metadata")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Tiling index: {tiling_index_path}")
    print(f"[INFO] Output CSV: {csv_path}")
    print(f"[INFO] Output Parquet: {parquet_path}")
    print(f"[INFO] Output GPKG: {gpkg_path}")
    print(f"[INFO] Summary CSV: {summary_path}")
    print(f"[INFO] Treat all-zero S2 as nodata: {args.treat_all_zero_s2_as_nodata}")
    print(f"[INFO] Nodata threshold percent: {args.nodata_threshold_percent}")
    print(f"[INFO] Cloud threshold percent: {args.cloud_threshold_percent}")

    tiling_df = load_tiling_index(
        path=tiling_index_path,
        selected_cities=args.city,
        max_patches=args.max_patches,
    )

    print(f"[INFO] Patch rows selected: {len(tiling_df)}")
    print(f"[INFO] Cities selected: {tiling_df['city'].nunique()}")
    print(f"[INFO] Splits present: {sorted(tiling_df['split'].unique().tolist())}")

    metadata_rows: List[Dict[str, Any]] = []

    for _, row in tqdm(tiling_df.iterrows(), total=len(tiling_df), desc="Computing patch metadata"):
        meta = compute_patch_metadata(
            output_root=output_root,
            row=row,
            treat_all_zero_s2_as_nodata=args.treat_all_zero_s2_as_nodata,
            nodata_threshold_percent=args.nodata_threshold_percent,
            cloud_threshold_percent=args.cloud_threshold_percent,
        )

        full_row = row.to_dict()
        full_row.update(meta)
        metadata_rows.append(full_row)

    df = pd.DataFrame(metadata_rows)

    df = df.sort_values(
        [
            "split_strategy",
            "fold_id",
            "split",
            "region_order",
            "city",
            "row_start",
            "col_start",
        ]
    ).reset_index(drop=True)

    df.to_csv(csv_path, index=False)

    parquet_written = write_parquet(df, parquet_path)

    summary_df = summarize_metadata(df)
    summary_df.to_csv(summary_path, index=False)

    write_markdown_summary(
        df=df,
        summary_df=summary_df,
        md_path=md_path,
        suffix=suffix,
        nodata_threshold_percent=args.nodata_threshold_percent,
        cloud_threshold_percent=args.cloud_threshold_percent,
    )

    gpkg_written = False

    if not args.no_gpkg:
        gpkg_written = write_gpkg(df, gpkg_path)

    print(f"[INFO] Wrote CSV: {csv_path}")

    if parquet_written:
        print(f"[INFO] Wrote Parquet: {parquet_path}")
    else:
        print("[INFO] Parquet not written.")

    print(f"[INFO] Wrote summary CSV: {summary_path}")
    print(f"[INFO] Wrote Markdown: {md_path}")

    if gpkg_written:
        print(f"[INFO] Wrote GeoPackage: {gpkg_path}")
    else:
        print("[INFO] GeoPackage not written.")

    total = len(df)
    positive = int(df["is_positive_patch"].sum())
    quality_pass = int(df["passes_basic_quality_filter"].sum())

    print("[INFO] Overall patch metadata summary:")
    print(f"       Total patches: {total}")
    print(f"       Positive patches: {positive}")
    print(f"       Negative patches: {total - positive}")
    print(f"       Positive patch percent: {100.0 * positive / total:.2f}%")
    print(f"       Quality-pass patches: {quality_pass}")
    print(f"       Quality-pass percent: {100.0 * quality_pass / total:.2f}%")

    print("[INFO] Summary by split:")
    print(
        df.groupby("split")
        .agg(
            patch_count=("patch_id", "count"),
            positive_patch_count=("is_positive_patch", "sum"),
            mean_label_positive_percent=("label_positive_percent", "mean"),
            mean_s2_nodata_percent=("s2_nodata_percent", "mean"),
            mean_s1_nodata_percent=("s1_nodata_percent", "mean"),
            quality_pass_count=("passes_basic_quality_filter", "sum"),
        )
        .reset_index()
        .to_string(index=False)
    )

    print("[INFO] Patch metadata computed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())