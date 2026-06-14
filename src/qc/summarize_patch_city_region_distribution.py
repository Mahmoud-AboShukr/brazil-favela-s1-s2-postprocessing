#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
summarize_patch_city_region_distribution.py

Purpose
-------
Summarise the spatial/administrative composition of the favela segmentation
dataset.

The script reads train.csv, val.csv, and test.csv from a dataset split directory
and reports:

    - Which regions are present.
    - Which cities are present in each region.
    - Patch counts per city and region.
    - Positive/favela patch counts per city and region.
    - Favela pixel totals and percentages.
    - Approximate favela area in hectares.
    - Split distribution across train/val/test.

Expected split CSV columns
--------------------------
At minimum:

    patch_id
    optical_path
    sar_path
    label_path
    row_start
    col_start
    city
    region

If the CSV contains label_positive_pixels, the script uses it directly.
If not, use --scan-labels to compute positive pixels by reading label raster windows.

Example
-------
python src\\qc\\summarize_patch_city_region_distribution.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --split-dir "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired/metadata/big_earth_net/same_city_low_holdout_ps224_st112_cover" `
  --patch-size 224 `
  --pixel-size-m 10 `
  --scan-labels
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    rasterio = None
    Window = None


POSITIVE_PIXEL_COLUMN_CANDIDATES = [
    "label_positive_pixels",
    "positive_pixels",
    "label_pos_pixels",
    "favela_pixels",
    "mask_positive_pixels",
    "pos_pixels",
    "n_positive_pixels",
]


REQUIRED_COLUMNS = [
    "patch_id",
    "label_path",
    "row_start",
    "col_start",
    "city",
    "region",
]


def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    log("ERROR", message)
    raise SystemExit(exit_code)


def warn(message: str) -> None:
    log("WARNING", message)


def path_to_str(path: Path) -> str:
    return str(path).replace("\\", "/")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def default_split_dir(instance_root: Path) -> Path:
    return (
        instance_root
        / "metadata"
        / "big_earth_net"
        / "same_city_low_holdout_ps224_st112_cover"
    )


def default_output_dir(instance_root: Path, split_dir: Path) -> Path:
    split_name = split_dir.name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        instance_root
        / "reports"
        / "qc"
        / f"patch_city_region_distribution_{split_name}_{stamp}"
    )


def resolve_path(path_value: Any, instance_root: Path) -> Path:
    raw = str(path_value).strip().replace("\\", "/")
    if raw == "":
        fail("Encountered empty path value in CSV.")

    p = Path(raw)

    if p.exists():
        return p

    if not p.is_absolute():
        candidate = instance_root / raw
        if candidate.exists():
            return candidate

    fail(
        "Path does not exist:\n"
        f"  original: {raw}\n"
        f"  tried:    {path_to_str(p)}"
    )


def find_positive_pixel_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested:
        if requested not in df.columns:
            fail(
                f"Requested positive pixel column does not exist: {requested}\n"
                f"Available columns: {list(df.columns)}"
            )
        return requested

    for col in POSITIVE_PIXEL_COLUMN_CANDIDATES:
        if col in df.columns:
            return col

    return None


def read_label_positive_pixels(
    label_path: Path,
    row_start: int,
    col_start: int,
    patch_size: int,
) -> int:
    if rasterio is None or Window is None:
        fail(
            "rasterio is required for --scan-labels, but it is not installed.\n"
            "Install it with:\n"
            "    pip install rasterio"
        )

    with rasterio.open(label_path) as src:
        window = Window(
            col_off=int(col_start),
            row_off=int(row_start),
            width=int(patch_size),
            height=int(patch_size),
        )

        arr = src.read(
            indexes=1,
            window=window,
            boundless=True,
            fill_value=0,
            out_shape=(patch_size, patch_size),
        )

    arr = np.nan_to_num(arr, nan=0, posinf=0, neginf=0)
    return int((arr > 0.5).sum())


def load_split_csv(
    split_name: str,
    csv_path: Path,
    instance_root: Path,
    patch_size: int,
    scan_labels: bool,
    positive_column: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not csv_path.exists():
        fail(f"{split_name}.csv does not exist:\n{path_to_str(csv_path)}")

    df = pd.read_csv(csv_path)

    if df.empty:
        fail(f"{split_name}.csv is empty:\n{path_to_str(csv_path)}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        fail(
            f"{split_name}.csv is missing required columns: {missing}\n"
            f"CSV: {path_to_str(csv_path)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["split"] = split_name
    df["city"] = df["city"].astype(str)
    df["region"] = df["region"].astype(str)
    df["row_start"] = pd.to_numeric(df["row_start"], errors="coerce").fillna(0).astype(int)
    df["col_start"] = pd.to_numeric(df["col_start"], errors="coerce").fillna(0).astype(int)

    found_positive_column = find_positive_pixel_column(df, positive_column)

    info: Dict[str, Any] = {
        "split": split_name,
        "csv_path": path_to_str(csv_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "positive_pixel_column": found_positive_column,
        "scan_labels": bool(scan_labels),
    }

    if found_positive_column is not None:
        df["label_positive_pixels_effective"] = (
            pd.to_numeric(df[found_positive_column], errors="coerce")
            .fillna(0)
            .clip(lower=0)
            .astype(np.int64)
        )
        info["positive_pixel_source"] = f"csv_column:{found_positive_column}"

    elif scan_labels:
        log("INFO", f"No positive-pixel column found in {split_name}. Scanning label rasters...")

        positive_pixels: List[int] = []

        for idx, row in df.iterrows():
            if idx % 500 == 0:
                log("INFO", f"{split_name}: scanned {idx:,}/{len(df):,} label windows")

            label_path = resolve_path(row["label_path"], instance_root)

            count = read_label_positive_pixels(
                label_path=label_path,
                row_start=int(row["row_start"]),
                col_start=int(row["col_start"]),
                patch_size=int(patch_size),
            )
            positive_pixels.append(count)

        df["label_positive_pixels_effective"] = np.asarray(positive_pixels, dtype=np.int64)
        info["positive_pixel_source"] = "scanned_label_windows"

    else:
        warn(
            f"No positive-pixel column found in {split_name}. "
            "The script will report patch counts, but favela-pixel statistics will be zero. "
            "Use --scan-labels for exact label statistics."
        )
        df["label_positive_pixels_effective"] = 0
        info["positive_pixel_source"] = "missing_fallback_zero"

    return df, info


def aggregate_distribution(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    patch_size: int,
    pixel_size_m: float,
    positive_min_pixels: int,
) -> pd.DataFrame:
    total_pixels_per_patch = int(patch_size) * int(patch_size)
    pixel_area_m2 = float(pixel_size_m) * float(pixel_size_m)

    working = df.copy()
    working["total_patch_pixels"] = total_pixels_per_patch
    working["is_positive_patch"] = (
        working["label_positive_pixels_effective"] >= int(positive_min_pixels)
    ).astype(int)

    grouped = working.groupby(list(group_cols), dropna=False)

    rows: List[Dict[str, Any]] = []

    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row: Dict[str, Any] = {
            col: key for col, key in zip(group_cols, keys)
        }

        patch_count = int(len(g))
        positive_patch_count = int(g["is_positive_patch"].sum())
        negative_patch_count = int(patch_count - positive_patch_count)

        favela_pixels = int(g["label_positive_pixels_effective"].sum())
        total_pixels = int(patch_count * total_pixels_per_patch)

        favela_pixel_pct = 100.0 * favela_pixels / total_pixels if total_pixels > 0 else 0.0
        positive_patch_pct = 100.0 * positive_patch_count / patch_count if patch_count > 0 else 0.0

        favela_area_ha = favela_pixels * pixel_area_m2 / 10000.0
        total_patch_area_ha = total_pixels * pixel_area_m2 / 10000.0

        patch_favela_pct_values = (
            100.0 * g["label_positive_pixels_effective"].astype(float) / total_pixels_per_patch
        )

        row.update(
            {
                "patch_count": patch_count,
                "positive_patch_count": positive_patch_count,
                "negative_patch_count": negative_patch_count,
                "positive_patch_pct": positive_patch_pct,
                "favela_pixels": favela_pixels,
                "total_pixels": total_pixels,
                "favela_pixel_pct": favela_pixel_pct,
                "approx_favela_area_ha": favela_area_ha,
                "approx_total_patch_area_ha": total_patch_area_ha,
                "mean_patch_favela_pct": float(patch_favela_pct_values.mean()),
                "median_patch_favela_pct": float(patch_favela_pct_values.median()),
                "std_patch_favela_pct": float(patch_favela_pct_values.std(ddof=0)),
                "max_patch_favela_pct": float(patch_favela_pct_values.max()),
            }
        )

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    sort_cols = list(group_cols) + ["patch_count"]
    sort_cols = [c for c in sort_cols if c in out.columns]

    return out.sort_values(sort_cols).reset_index(drop=True)


def add_dataset_share_columns(df: pd.DataFrame, total_patches: int, total_favela_pixels: int) -> pd.DataFrame:
    out = df.copy()

    if "patch_count" in out.columns:
        out["patch_share_of_dataset_pct"] = (
            100.0 * out["patch_count"] / max(1, int(total_patches))
        )

    if "favela_pixels" in out.columns:
        out["favela_pixel_share_of_dataset_pct"] = (
            100.0 * out["favela_pixels"] / max(1, int(total_favela_pixels))
            if total_favela_pixels > 0
            else 0.0
        )

    return out


def make_city_region_matrix(city_total: pd.DataFrame) -> pd.DataFrame:
    if city_total.empty:
        return city_total

    cols = [
        "region",
        "city",
        "patch_count",
        "positive_patch_count",
        "positive_patch_pct",
        "favela_pixel_pct",
        "approx_favela_area_ha",
        "patch_share_of_dataset_pct",
        "favela_pixel_share_of_dataset_pct",
    ]

    cols = [c for c in cols if c in city_total.columns]

    return (
        city_total[cols]
        .sort_values(["region", "city"])
        .reset_index(drop=True)
    )


def safe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"

    small = df.head(max_rows).copy()

    try:
        return small.to_markdown(index=False)
    except Exception:
        return "```\n" + small.to_string(index=False) + "\n```"


def write_markdown_report(
    path: Path,
    dataset_summary: Dict[str, Any],
    region_total: pd.DataFrame,
    city_total: pd.DataFrame,
    region_split: pd.DataFrame,
    city_split: pd.DataFrame,
) -> None:
    ensure_dir(path.parent)

    top_regions = region_total.sort_values("patch_count", ascending=False)
    top_cities_by_patches = city_total.sort_values("patch_count", ascending=False)
    top_cities_by_favela_area = city_total.sort_values("approx_favela_area_ha", ascending=False)
    top_cities_by_favela_pct = city_total.sort_values("favela_pixel_pct", ascending=False)

    text = f"""# Patch, City, Region, and Favela Distribution Summary

Generated: `{dataset_summary["created_at"]}`

## Dataset summary

- Split directory: `{dataset_summary["split_dir"]}`
- Total patches: `{dataset_summary["total_patches"]:,}`
- Positive patches: `{dataset_summary["positive_patches"]:,}`
- Negative patches: `{dataset_summary["negative_patches"]:,}`
- Positive patch percentage: `{dataset_summary["positive_patch_pct"]:.3f}%`
- Total favela pixels: `{dataset_summary["total_favela_pixels"]:,}`
- Favela pixel percentage: `{dataset_summary["favela_pixel_pct"]:.4f}%`
- Approximate favela area: `{dataset_summary["approx_favela_area_ha"]:.2f} ha`
- Number of regions: `{dataset_summary["n_regions"]}`
- Number of cities: `{dataset_summary["n_cities"]}`

## Regions by patch count

{safe_to_markdown(top_regions, max_rows=20)}

## Cities by patch count

{safe_to_markdown(top_cities_by_patches, max_rows=30)}

## Cities by approximate favela area

{safe_to_markdown(top_cities_by_favela_area, max_rows=30)}

## Cities by favela pixel percentage

{safe_to_markdown(top_cities_by_favela_pct, max_rows=30)}

## Region split distribution

{safe_to_markdown(region_split, max_rows=50)}

## City split distribution

{safe_to_markdown(city_split, max_rows=80)}

## Interpretation guide

- `patch_count`: number of generated patches for that city/region.
- `positive_patch_count`: number of patches containing at least the chosen number of favela pixels.
- `positive_patch_pct`: percentage of patches that contain favela pixels.
- `favela_pixels`: total labelled favela pixels across patches.
- `favela_pixel_pct`: favela pixels divided by total patch pixels.
- `approx_favela_area_ha`: approximate favela-labelled area in hectares, assuming the pixel size given to the script.
- `patch_share_of_dataset_pct`: how much of the dataset's patch count comes from the city/region.
- `favela_pixel_share_of_dataset_pct`: how much of the dataset's favela-labelled pixels come from the city/region.

Important: because patches can overlap, `favela_pixels` and `approx_favela_area_ha`
represent patch-level training distribution statistics, not a unique non-overlapping
map-area measurement.
"""

    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    instance_root = Path(args.instance_root)
    split_dir = Path(args.split_dir) if args.split_dir else default_split_dir(instance_root)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(instance_root, split_dir)

    ensure_dir(output_dir)

    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Split dir:     {path_to_str(split_dir)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")

    split_infos: List[Dict[str, Any]] = []
    split_dfs: List[pd.DataFrame] = []

    for split_name in args.splits:
        csv_path = split_dir / f"{split_name}.csv"
        df, info = load_split_csv(
            split_name=split_name,
            csv_path=csv_path,
            instance_root=instance_root,
            patch_size=int(args.patch_size),
            scan_labels=bool(args.scan_labels),
            positive_column=args.positive_column,
        )

        split_dfs.append(df)
        split_infos.append(info)

    all_df = pd.concat(split_dfs, ignore_index=True)

    total_pixels_per_patch = int(args.patch_size) * int(args.patch_size)
    pixel_area_m2 = float(args.pixel_size_m) * float(args.pixel_size_m)

    all_df["is_positive_patch"] = (
        all_df["label_positive_pixels_effective"] >= int(args.positive_min_pixels)
    ).astype(int)

    all_df["patch_favela_pct"] = (
        100.0
        * all_df["label_positive_pixels_effective"].astype(float)
        / float(total_pixels_per_patch)
    )

    all_df["approx_patch_favela_area_ha"] = (
        all_df["label_positive_pixels_effective"].astype(float)
        * pixel_area_m2
        / 10000.0
    )

    total_patches = int(len(all_df))
    total_positive_patches = int(all_df["is_positive_patch"].sum())
    total_negative_patches = int(total_patches - total_positive_patches)
    total_favela_pixels = int(all_df["label_positive_pixels_effective"].sum())
    total_pixels = int(total_patches * total_pixels_per_patch)

    dataset_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "instance_root": path_to_str(instance_root),
        "split_dir": path_to_str(split_dir),
        "output_dir": path_to_str(output_dir),
        "splits": list(args.splits),
        "patch_size": int(args.patch_size),
        "pixel_size_m": float(args.pixel_size_m),
        "positive_min_pixels": int(args.positive_min_pixels),
        "total_patches": total_patches,
        "positive_patches": total_positive_patches,
        "negative_patches": total_negative_patches,
        "positive_patch_pct": 100.0 * total_positive_patches / max(1, total_patches),
        "total_favela_pixels": total_favela_pixels,
        "total_pixels": total_pixels,
        "favela_pixel_pct": 100.0 * total_favela_pixels / max(1, total_pixels),
        "approx_favela_area_ha": total_favela_pixels * pixel_area_m2 / 10000.0,
        "n_regions": int(all_df["region"].nunique()),
        "n_cities": int(all_df["city"].nunique()),
        "split_infos": split_infos,
    }

    split_summary = aggregate_distribution(
        all_df,
        group_cols=["split"],
        patch_size=int(args.patch_size),
        pixel_size_m=float(args.pixel_size_m),
        positive_min_pixels=int(args.positive_min_pixels),
    )

    region_total = aggregate_distribution(
        all_df,
        group_cols=["region"],
        patch_size=int(args.patch_size),
        pixel_size_m=float(args.pixel_size_m),
        positive_min_pixels=int(args.positive_min_pixels),
    )

    region_split = aggregate_distribution(
        all_df,
        group_cols=["region", "split"],
        patch_size=int(args.patch_size),
        pixel_size_m=float(args.pixel_size_m),
        positive_min_pixels=int(args.positive_min_pixels),
    )

    city_total = aggregate_distribution(
        all_df,
        group_cols=["region", "city"],
        patch_size=int(args.patch_size),
        pixel_size_m=float(args.pixel_size_m),
        positive_min_pixels=int(args.positive_min_pixels),
    )

    city_split = aggregate_distribution(
        all_df,
        group_cols=["region", "city", "split"],
        patch_size=int(args.patch_size),
        pixel_size_m=float(args.pixel_size_m),
        positive_min_pixels=int(args.positive_min_pixels),
    )

    region_total = add_dataset_share_columns(region_total, total_patches, total_favela_pixels)
    region_split = add_dataset_share_columns(region_split, total_patches, total_favela_pixels)
    city_total = add_dataset_share_columns(city_total, total_patches, total_favela_pixels)
    city_split = add_dataset_share_columns(city_split, total_patches, total_favela_pixels)

    city_region_matrix = make_city_region_matrix(city_total)

    all_df.to_csv(output_dir / "patch_level_distribution_with_effective_positive_pixels.csv", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    region_total.to_csv(output_dir / "region_summary_all_splits.csv", index=False)
    region_split.to_csv(output_dir / "region_summary_by_split.csv", index=False)
    city_total.to_csv(output_dir / "city_summary_all_splits.csv", index=False)
    city_split.to_csv(output_dir / "city_summary_by_split.csv", index=False)
    city_region_matrix.to_csv(output_dir / "city_region_matrix_for_report.csv", index=False)

    top_cities_by_patch_count = city_total.sort_values("patch_count", ascending=False)
    top_cities_by_favela_pixels = city_total.sort_values("favela_pixels", ascending=False)
    top_cities_by_favela_pct = city_total.sort_values("favela_pixel_pct", ascending=False)

    top_cities_by_patch_count.to_csv(output_dir / "top_cities_by_patch_count.csv", index=False)
    top_cities_by_favela_pixels.to_csv(output_dir / "top_cities_by_favela_pixels.csv", index=False)
    top_cities_by_favela_pct.to_csv(output_dir / "top_cities_by_favela_pixel_pct.csv", index=False)

    write_json(output_dir / "dataset_distribution_summary.json", dataset_summary)

    write_markdown_report(
        path=output_dir / "patch_city_region_distribution_report.md",
        dataset_summary=dataset_summary,
        region_total=region_total,
        city_total=city_total,
        region_split=region_split,
        city_split=city_split,
    )

    log("OK", "Distribution summaries written.")
    log("OK", f"Output directory:\n{path_to_str(output_dir)}")
    log("OK", f"Total patches: {dataset_summary['total_patches']:,}")
    log("OK", f"Regions: {dataset_summary['n_regions']:,}")
    log("OK", f"Cities: {dataset_summary['n_cities']:,}")
    log("OK", f"Positive patches: {dataset_summary['positive_patches']:,}")
    log("OK", f"Favela pixel percentage: {dataset_summary['favela_pixel_pct']:.4f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise patch, city, region, and favela distribution for the dataset."
    )

    parser.add_argument(
        "--instance-root",
        required=True,
        help="Dataset instance root.",
    )

    parser.add_argument(
        "--split-dir",
        default=None,
        help=(
            "Directory containing train.csv, val.csv, and test.csv. "
            "If omitted, uses metadata/big_earth_net/same_city_low_holdout_ps224_st112_cover."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for CSV summaries. If omitted, writes under instance_root/reports/qc/.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split names to read from split_dir. Default: train val test.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size in pixels. Default: 224.",
    )

    parser.add_argument(
        "--pixel-size-m",
        type=float,
        default=10.0,
        help="Pixel size in metres. Default: 10 for Sentinel-aligned 10 m labels.",
    )

    parser.add_argument(
        "--positive-min-pixels",
        type=int,
        default=1,
        help="Minimum number of favela pixels for a patch to count as positive. Default: 1.",
    )

    parser.add_argument(
        "--positive-column",
        default=None,
        help=(
            "Optional explicit positive-pixel column. "
            "If omitted, common names such as label_positive_pixels are detected automatically."
        ),
    )

    parser.add_argument(
        "--scan-labels",
        action="store_true",
        help=(
            "If no positive-pixel column exists, scan label rasters to compute positive pixels. "
            "This is slower but exact."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())