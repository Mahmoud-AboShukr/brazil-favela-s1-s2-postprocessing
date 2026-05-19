#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_compute_patch_metadata_224.py

Compute patch-level metadata for the 224x224 instance C patch tiling index.

This script reads the patch index created by:

    src/instance_C/01_build_patch_tiling_index_224.py

It does NOT export physical patch images.

For every patch, it reads the corresponding 224x224 windows from:

    - S2 filled raster
    - S1 SNAP-GRD ready raster
    - label raster
    - optional future S1 RTC raster, if present

Then it computes metadata needed by both:

    1. CROMA representation/probing experiments
    2. CROMA + UPerNet segmentation split-strategy experiments

Main outputs:

    metadata/instance_C_patches/patch_metadata_ps224_st112_cover.csv
    metadata/instance_C_patches/patch_metadata_ps224_st112_cover.json
    metadata/instance_C_patches/patch_metadata_ps224_st112_cover.md

Example PowerShell command:

python src/instance_C/02_compute_patch_metadata_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --s2-all-zero-as-nodata `
  --no-s1-all-zero-as-nodata `
  --nan-as-nodata `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
except ImportError as exc:
    raise SystemExit(
        "[ERROR] rasterio is required but is not installed.\n"
        "Install it first, for example:\n"
        "    pip install rasterio\n\n"
        f"Original error: {exc}"
    )


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


def str_to_path(value: object) -> Optional[Path]:
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    return Path(text)


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)


def parse_int(row: Dict[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except Exception as exc:
        raise ValueError(
            f"Could not parse integer column '{key}' from row with patch_id="
            f"{row.get('patch_id', '<unknown>')}"
        ) from exc


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
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


def write_csv(path: Path, rows: List[Dict[str, object]], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    if not rows:
        fail("No rows generated. Refusing to write empty CSV.")

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(path: Path, summary: Dict[str, object], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Instance C patch metadata")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Input tiling index: `{summary['input_tiling_index']}`")
    lines.append(f"- Output metadata CSV: `{summary['outputs']['csv']}`")
    lines.append(f"- Patch size: `{summary['parameters']['patch_size']}`")
    lines.append(f"- Stride: `{summary['parameters']['stride']}`")
    lines.append(f"- Edge mode: `{summary['parameters']['edge_mode']}`")
    lines.append(f"- Total patches: `{summary['total_patches']}`")
    lines.append(f"- Positive patches: `{summary['positive_patches']}`")
    lines.append(f"- Positive patch percent: `{summary['positive_patch_percent']:.6f}`")
    lines.append(f"- Empty patches: `{summary['empty_patches']}`")
    lines.append(f"- Empty patch percent: `{summary['empty_patch_percent']:.6f}`")
    lines.append(f"- Total label positive pixels: `{summary['total_label_positive_pixels']}`")
    lines.append(f"- Label non-binary patches: `{summary['label_non_binary_patches']}`")
    lines.append(f"- Label non-binary pixels: `{summary['label_non_binary_pixels']}`")
    lines.append(f"- Patches with S2 valid percent < 100: `{summary['patches_s2_valid_lt_100']}`")
    lines.append(f"- Patches with S1 SNAP-GRD valid percent < 100: `{summary['patches_s1_snap_grd_valid_lt_100']}`")
    lines.append(f"- Patches with S1 RTC available: `{summary['patches_with_s1_rtc']}`")
    lines.append("")

    lines.append("## Label density bins")
    lines.append("")
    lines.append("| density bin | patches |")
    lines.append("|---|---:|")

    for key, value in summary["label_density_bin_counts"].items():
        lines.append(f"| {key} | {value} |")

    lines.append("")
    lines.append("## Metadata by region")
    lines.append("")
    lines.append(
        "| region | patches | positive patches | positive patch % | "
        "mean label positive % | max label positive % | min S2 valid % | min S1 valid % |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for region, item in summary["by_region"].items():
        lines.append(
            f"| {region} | "
            f"{item['patches']} | "
            f"{item['positive_patches']} | "
            f"{item['positive_patch_percent']:.6f} | "
            f"{item['mean_label_positive_percent']:.6f} | "
            f"{item['max_label_positive_percent']:.6f} | "
            f"{item['min_s2_valid_percent']:.6f} | "
            f"{item['min_s1_snap_grd_valid_percent']:.6f} |"
        )

    lines.append("")
    lines.append("## Metadata by city")
    lines.append("")
    lines.append(
        "| city | region | patches | positive patches | positive patch % | "
        "mean label positive % | max label positive % | min S2 valid % | min S1 valid % |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for city, item in summary["by_city"].items():
        lines.append(
            f"| {city} | "
            f"{item['region']} | "
            f"{item['patches']} | "
            f"{item['positive_patches']} | "
            f"{item['positive_patch_percent']:.6f} | "
            f"{item['mean_label_positive_percent']:.6f} | "
            f"{item['max_label_positive_percent']:.6f} | "
            f"{item['min_s2_valid_percent']:.6f} | "
            f"{item['min_s1_snap_grd_valid_percent']:.6f} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `patch_label_binary = 1` means the patch contains at least one favela label pixel.")
    lines.append("- `label_positive_percent` is the percentage of pixels inside the 224x224 patch labeled as favela.")
    lines.append("- `label_density_bin` is a diagnostic bin, not a final scientific class.")
    lines.append("- `s2_valid_percent` and `s1_snap_grd_valid_percent` should ideally be 100 after the nodata repair workflow.")
    lines.append("- `s1_rtc_*` fields are blank until `s1_rtc_ready/` is created.")
    lines.append("- This table is the common input for CROMA embedding/probing and CROMA+UPerNet segmentation split diagnostics.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Mask / validity helpers
# ---------------------------------------------------------------------

def masked_array_to_data_and_mask(array: np.ma.MaskedArray) -> Tuple[np.ndarray, np.ndarray]:
    data = np.ma.getdata(array)
    mask = np.ma.getmaskarray(array)

    if mask.shape == ():
        mask = np.zeros(data.shape, dtype=bool)

    return data, mask


def compute_multiband_validity(
    array: np.ma.MaskedArray,
    *,
    all_zero_as_nodata: bool,
    nan_as_nodata: bool,
) -> Tuple[int, int, float]:
    """
    Compute valid pixels for a multiband raster window.

    Shape expected:
        bands x height x width

    Pixel is invalid if:
        - it is masked in any band
        - any band is NaN/Inf when nan_as_nodata=True
        - all bands are zero when all_zero_as_nodata=True
    """

    data, mask = masked_array_to_data_and_mask(array)

    if data.ndim != 3:
        fail(f"Expected multiband array with shape (bands, height, width), got {data.shape}")

    _, height, width = data.shape
    total_pixels = height * width

    invalid = np.any(mask, axis=0)

    if nan_as_nodata:
        invalid |= ~np.all(np.isfinite(data), axis=0)

    if all_zero_as_nodata:
        invalid |= np.all(data == 0, axis=0)

    invalid_pixels = int(np.count_nonzero(invalid))
    valid_pixels = int(total_pixels - invalid_pixels)
    valid_percent = 100.0 * valid_pixels / total_pixels

    return valid_pixels, invalid_pixels, valid_percent


def compute_label_metadata(label_array: np.ma.MaskedArray) -> Dict[str, object]:
    """
    Compute label statistics for one patch.

    Label is expected to be binary 0/1.
    Positive means valid pixel value > 0.
    """

    data, mask = masked_array_to_data_and_mask(label_array)

    if data.ndim != 2:
        fail(f"Expected label array with shape (height, width), got {data.shape}")

    height, width = data.shape
    total_pixels = height * width

    valid = ~mask

    if np.issubdtype(data.dtype, np.floating):
        valid &= np.isfinite(data)

    valid_pixels = int(np.count_nonzero(valid))
    invalid_pixels = int(total_pixels - valid_pixels)
    valid_percent = 100.0 * valid_pixels / total_pixels

    positive = valid & (data > 0)
    positive_pixels = int(np.count_nonzero(positive))
    positive_percent = 100.0 * positive_pixels / total_pixels

    valid_values = data[valid]

    if valid_values.size == 0:
        non_binary_pixels = 0
    else:
        binary_mask = (valid_values == 0) | (valid_values == 1)
        non_binary_pixels = int(np.count_nonzero(~binary_mask))

    patch_label_binary = 1 if positive_pixels > 0 else 0

    return {
        "label_valid_pixels": valid_pixels,
        "label_invalid_pixels": invalid_pixels,
        "label_valid_percent": valid_percent,
        "label_positive_pixels": positive_pixels,
        "label_positive_percent": positive_percent,
        "has_positive_label": bool(positive_pixels > 0),
        "patch_label_binary": patch_label_binary,
        "label_non_binary_pixels": non_binary_pixels,
    }


def get_density_bin(
    label_positive_percent: float,
    low_threshold_percent: float,
    high_threshold_percent: float,
) -> str:
    if label_positive_percent <= 0.0:
        return "empty"

    if label_positive_percent < low_threshold_percent:
        return "low"

    if label_positive_percent < high_threshold_percent:
        return "medium"

    return "high"


# ---------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------

def init_group_stats() -> Dict[str, object]:
    return {
        "patches": 0,
        "positive_patches": 0,
        "total_label_positive_pixels": 0,
        "label_positive_percent_sum": 0.0,
        "max_label_positive_percent": 0.0,
        "min_s2_valid_percent": 100.0,
        "min_s1_snap_grd_valid_percent": 100.0,
        "s2_valid_percent_sum": 0.0,
        "s1_snap_grd_valid_percent_sum": 0.0,
        "label_non_binary_pixels": 0,
        "label_non_binary_patches": 0,
    }


def update_group_stats(stats: Dict[str, object], row: Dict[str, object]) -> None:
    label_positive_percent = float(row["label_positive_percent"])
    s2_valid_percent = float(row["s2_valid_percent"])
    s1_valid_percent = float(row["s1_snap_grd_valid_percent"])
    label_positive_pixels = int(row["label_positive_pixels"])
    patch_label_binary = int(row["patch_label_binary"])
    non_binary_pixels = int(row["label_non_binary_pixels"])

    stats["patches"] = int(stats["patches"]) + 1
    stats["positive_patches"] = int(stats["positive_patches"]) + patch_label_binary
    stats["total_label_positive_pixels"] = int(stats["total_label_positive_pixels"]) + label_positive_pixels

    stats["label_positive_percent_sum"] = (
        float(stats["label_positive_percent_sum"]) + label_positive_percent
    )

    stats["max_label_positive_percent"] = max(
        float(stats["max_label_positive_percent"]),
        label_positive_percent,
    )

    stats["min_s2_valid_percent"] = min(
        float(stats["min_s2_valid_percent"]),
        s2_valid_percent,
    )

    stats["min_s1_snap_grd_valid_percent"] = min(
        float(stats["min_s1_snap_grd_valid_percent"]),
        s1_valid_percent,
    )

    stats["s2_valid_percent_sum"] = (
        float(stats["s2_valid_percent_sum"]) + s2_valid_percent
    )

    stats["s1_snap_grd_valid_percent_sum"] = (
        float(stats["s1_snap_grd_valid_percent_sum"]) + s1_valid_percent
    )

    stats["label_non_binary_pixels"] = (
        int(stats["label_non_binary_pixels"]) + non_binary_pixels
    )

    if non_binary_pixels > 0:
        stats["label_non_binary_patches"] = int(stats["label_non_binary_patches"]) + 1


def finalize_group_stats(stats: Dict[str, object], region: Optional[str] = None) -> Dict[str, object]:
    patches = int(stats["patches"])
    positive_patches = int(stats["positive_patches"])

    if patches == 0:
        positive_patch_percent = 0.0
        mean_label_positive_percent = 0.0
        mean_s2_valid_percent = 0.0
        mean_s1_valid_percent = 0.0
    else:
        positive_patch_percent = 100.0 * positive_patches / patches
        mean_label_positive_percent = float(stats["label_positive_percent_sum"]) / patches
        mean_s2_valid_percent = float(stats["s2_valid_percent_sum"]) / patches
        mean_s1_valid_percent = float(stats["s1_snap_grd_valid_percent_sum"]) / patches

    result = {
        "patches": patches,
        "positive_patches": positive_patches,
        "positive_patch_percent": positive_patch_percent,
        "total_label_positive_pixels": int(stats["total_label_positive_pixels"]),
        "mean_label_positive_percent": mean_label_positive_percent,
        "max_label_positive_percent": float(stats["max_label_positive_percent"]),
        "min_s2_valid_percent": float(stats["min_s2_valid_percent"]),
        "mean_s2_valid_percent": mean_s2_valid_percent,
        "min_s1_snap_grd_valid_percent": float(stats["min_s1_snap_grd_valid_percent"]),
        "mean_s1_snap_grd_valid_percent": mean_s1_valid_percent,
        "label_non_binary_pixels": int(stats["label_non_binary_pixels"]),
        "label_non_binary_patches": int(stats["label_non_binary_patches"]),
    }

    if region is not None:
        result["region"] = region

    return result


# ---------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------

def validate_required_columns(rows: List[Dict[str, str]]) -> None:
    required = [
        "patch_id",
        "city",
        "region",
        "row_start",
        "col_start",
        "height",
        "width",
        "source_s2_path",
        "source_s1_snap_grd_path",
        "source_s1_rtc_path",
        "source_label_path",
    ]

    columns = set(rows[0].keys())
    missing = [col for col in required if col not in columns]

    if missing:
        fail(
            "Input tiling index is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def group_rows_by_city(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        city = row.get("city", "").strip()

        if city == "":
            fail(f"Found row with empty city: {row}")

        grouped[city].append(row)

    return dict(sorted(grouped.items()))


def process_city_rows(
    city: str,
    city_rows: List[Dict[str, str]],
    *,
    s2_all_zero_as_nodata: bool,
    s1_all_zero_as_nodata: bool,
    s1_rtc_all_zero_as_nodata: bool,
    nan_as_nodata: bool,
    low_density_threshold_percent: float,
    high_density_threshold_percent: float,
    progress_every: int,
) -> List[Dict[str, object]]:
    first = city_rows[0]

    s2_path = str_to_path(first["source_s2_path"])
    s1_path = str_to_path(first["source_s1_snap_grd_path"])
    label_path = str_to_path(first["source_label_path"])
    s1_rtc_path = str_to_path(first.get("source_s1_rtc_path", ""))

    if s2_path is None or not s2_path.exists():
        fail(f"{city}: missing S2 raster: {path_to_str(s2_path)}")

    if s1_path is None or not s1_path.exists():
        fail(f"{city}: missing S1 SNAP-GRD raster: {path_to_str(s1_path)}")

    if label_path is None or not label_path.exists():
        fail(f"{city}: missing label raster: {path_to_str(label_path)}")

    s1_rtc_available = s1_rtc_path is not None and s1_rtc_path.exists()

    metadata_rows: List[Dict[str, object]] = []

    with rasterio.open(s2_path) as s2_src, rasterio.open(s1_path) as s1_src, rasterio.open(label_path) as label_src:
        if s1_rtc_available:
            s1_rtc_src = rasterio.open(s1_rtc_path)
        else:
            s1_rtc_src = None

        try:
            for idx, row in enumerate(city_rows, start=1):
                row_start = parse_int(row, "row_start")
                col_start = parse_int(row, "col_start")
                height = parse_int(row, "height")
                width = parse_int(row, "width")

                window = Window(
                    col_off=col_start,
                    row_off=row_start,
                    width=width,
                    height=height,
                )

                label_patch = label_src.read(1, window=window, masked=True)
                label_meta = compute_label_metadata(label_patch)

                s2_patch = s2_src.read(window=window, masked=True)
                s2_valid_pixels, s2_invalid_pixels, s2_valid_percent = compute_multiband_validity(
                    s2_patch,
                    all_zero_as_nodata=s2_all_zero_as_nodata,
                    nan_as_nodata=nan_as_nodata,
                )

                s1_patch = s1_src.read(window=window, masked=True)
                s1_valid_pixels, s1_invalid_pixels, s1_valid_percent = compute_multiband_validity(
                    s1_patch,
                    all_zero_as_nodata=s1_all_zero_as_nodata,
                    nan_as_nodata=nan_as_nodata,
                )

                if s1_rtc_src is not None:
                    s1_rtc_patch = s1_rtc_src.read(window=window, masked=True)
                    s1_rtc_valid_pixels, s1_rtc_invalid_pixels, s1_rtc_valid_percent = compute_multiband_validity(
                        s1_rtc_patch,
                        all_zero_as_nodata=s1_rtc_all_zero_as_nodata,
                        nan_as_nodata=nan_as_nodata,
                    )
                    s1_rtc_exists = True
                else:
                    s1_rtc_valid_pixels = ""
                    s1_rtc_invalid_pixels = ""
                    s1_rtc_valid_percent = ""
                    s1_rtc_exists = False

                label_positive_percent = float(label_meta["label_positive_percent"])

                density_bin = get_density_bin(
                    label_positive_percent=label_positive_percent,
                    low_threshold_percent=low_density_threshold_percent,
                    high_threshold_percent=high_density_threshold_percent,
                )

                patch_label_binary = int(label_meta["patch_label_binary"])

                output_row: Dict[str, object] = dict(row)

                output_row.update(
                    {
                        "label_valid_pixels": int(label_meta["label_valid_pixels"]),
                        "label_invalid_pixels": int(label_meta["label_invalid_pixels"]),
                        "label_valid_percent": round(float(label_meta["label_valid_percent"]), 8),
                        "label_positive_pixels": int(label_meta["label_positive_pixels"]),
                        "label_positive_percent": round(label_positive_percent, 8),
                        "has_positive_label": bool(label_meta["has_positive_label"]),
                        "patch_label_binary": patch_label_binary,
                        "croma_patch_label": patch_label_binary,
                        "upernet_patch_has_positive_pixels": patch_label_binary,
                        "label_non_binary_pixels": int(label_meta["label_non_binary_pixels"]),
                        "label_density_bin": density_bin,
                        "s2_valid_pixels": int(s2_valid_pixels),
                        "s2_invalid_pixels": int(s2_invalid_pixels),
                        "s2_valid_percent": round(float(s2_valid_percent), 8),
                        "s1_snap_grd_valid_pixels": int(s1_valid_pixels),
                        "s1_snap_grd_invalid_pixels": int(s1_invalid_pixels),
                        "s1_snap_grd_valid_percent": round(float(s1_valid_percent), 8),
                        "s1_rtc_exists": bool(s1_rtc_exists),
                        "s1_rtc_valid_pixels": s1_rtc_valid_pixels,
                        "s1_rtc_invalid_pixels": s1_rtc_invalid_pixels,
                        "s1_rtc_valid_percent": (
                            round(float(s1_rtc_valid_percent), 8)
                            if s1_rtc_valid_percent != ""
                            else ""
                        ),
                    }
                )

                metadata_rows.append(output_row)

                if progress_every > 0 and idx % progress_every == 0:
                    log("INFO", f"{city}: processed {idx}/{len(city_rows)} patches")

        finally:
            if s1_rtc_src is not None:
                s1_rtc_src.close()

    return metadata_rows


def build_summary(
    metadata_rows: List[Dict[str, object]],
    *,
    instance_root: Path,
    input_csv: Path,
    output_csv: Path,
    output_json: Path,
    output_md: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    total_patches = len(metadata_rows)

    positive_patches = sum(int(row["patch_label_binary"]) for row in metadata_rows)
    empty_patches = total_patches - positive_patches

    positive_patch_percent = 100.0 * positive_patches / total_patches if total_patches else 0.0
    empty_patch_percent = 100.0 * empty_patches / total_patches if total_patches else 0.0

    total_label_positive_pixels = sum(int(row["label_positive_pixels"]) for row in metadata_rows)

    label_non_binary_pixels = sum(int(row["label_non_binary_pixels"]) for row in metadata_rows)

    label_non_binary_patches = sum(
        1 for row in metadata_rows
        if int(row["label_non_binary_pixels"]) > 0
    )

    patches_s2_valid_lt_100 = sum(
        1 for row in metadata_rows
        if float(row["s2_valid_percent"]) < 100.0
    )

    patches_s1_snap_grd_valid_lt_100 = sum(
        1 for row in metadata_rows
        if float(row["s1_snap_grd_valid_percent"]) < 100.0
    )

    patches_with_s1_rtc = sum(
        1 for row in metadata_rows
        if bool(row["s1_rtc_exists"])
    )

    density_counter = Counter(str(row["label_density_bin"]) for row in metadata_rows)

    density_counts = {
        "empty": density_counter.get("empty", 0),
        "low": density_counter.get("low", 0),
        "medium": density_counter.get("medium", 0),
        "high": density_counter.get("high", 0),
    }

    city_stats_raw: Dict[str, Dict[str, object]] = defaultdict(init_group_stats)
    region_stats_raw: Dict[str, Dict[str, object]] = defaultdict(init_group_stats)
    city_to_region: Dict[str, str] = {}

    for row in metadata_rows:
        city = str(row["city"])
        region = str(row["region"])

        city_to_region[city] = region

        update_group_stats(city_stats_raw[city], row)
        update_group_stats(region_stats_raw[region], row)

    by_city: Dict[str, Dict[str, object]] = {}

    for city in sorted(city_stats_raw):
        by_city[city] = finalize_group_stats(
            city_stats_raw[city],
            region=city_to_region.get(city, ""),
        )

    by_region: Dict[str, Dict[str, object]] = {}

    for region in sorted(region_stats_raw):
        by_region[region] = finalize_group_stats(region_stats_raw[region])

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "input_tiling_index": path_to_str(input_csv),
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "s2_all_zero_as_nodata": bool(args.s2_all_zero_as_nodata),
            "s1_all_zero_as_nodata": bool(args.s1_all_zero_as_nodata),
            "s1_rtc_all_zero_as_nodata": bool(args.s1_rtc_all_zero_as_nodata),
            "nan_as_nodata": bool(args.nan_as_nodata),
            "low_density_threshold_percent": float(args.low_density_threshold_percent),
            "high_density_threshold_percent": float(args.high_density_threshold_percent),
        },
        "total_patches": total_patches,
        "positive_patches": positive_patches,
        "positive_patch_percent": positive_patch_percent,
        "empty_patches": empty_patches,
        "empty_patch_percent": empty_patch_percent,
        "total_label_positive_pixels": int(total_label_positive_pixels),
        "label_non_binary_pixels": int(label_non_binary_pixels),
        "label_non_binary_patches": int(label_non_binary_patches),
        "patches_s2_valid_lt_100": int(patches_s2_valid_lt_100),
        "patches_s1_snap_grd_valid_lt_100": int(patches_s1_snap_grd_valid_lt_100),
        "patches_with_s1_rtc": int(patches_with_s1_rtc),
        "label_density_bin_counts": density_counts,
        "by_city": by_city,
        "by_region": by_region,
        "outputs": {
            "csv": path_to_str(output_csv),
            "json": path_to_str(output_json),
            "markdown": path_to_str(output_md),
        },
    }

    return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute patch-level metadata for the instance C 224x224 patch index."
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
        help="Patch size used by the tiling index. Default: 224.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=112,
        help="Stride used by the tiling index. Default: 112.",
    )

    parser.add_argument(
        "--edge-mode",
        choices=["cover", "drop"],
        default="cover",
        help="Edge mode used by the tiling index. Default: cover.",
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to the patch tiling CSV. "
            "Default: <instance-root>/metadata/instance_C_patches/"
            "patch_tiling_index_ps<patch-size>_st<stride>_<edge-mode>.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. "
            "Default: <instance-root>/metadata/instance_C_patches"
        ),
    )

    parser.add_argument(
        "--s2-all-zero-as-nodata",
        action="store_true",
        help="Treat S2 pixels where all bands are zero as nodata.",
    )

    parser.add_argument(
        "--s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_true",
        help="Treat S1 SNAP-GRD pixels where all bands are zero as nodata.",
    )

    parser.add_argument(
        "--no-s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_false",
        help="Do not treat S1 SNAP-GRD all-zero pixels as nodata. Recommended here.",
    )

    parser.set_defaults(s1_all_zero_as_nodata=False)

    parser.add_argument(
        "--s1-rtc-all-zero-as-nodata",
        dest="s1_rtc_all_zero_as_nodata",
        action="store_true",
        help="Treat S1 RTC pixels where all bands are zero as nodata.",
    )

    parser.add_argument(
        "--no-s1-rtc-all-zero-as-nodata",
        dest="s1_rtc_all_zero_as_nodata",
        action="store_false",
        help="Do not treat S1 RTC all-zero pixels as nodata.",
    )

    parser.set_defaults(s1_rtc_all_zero_as_nodata=False)

    parser.add_argument(
        "--nan-as-nodata",
        action="store_true",
        help="Treat NaN/Inf values as nodata.",
    )

    parser.add_argument(
        "--low-density-threshold-percent",
        type=float,
        default=1.0,
        help="Threshold between low and medium label density. Default: 1.0.",
    )

    parser.add_argument(
        "--high-density-threshold-percent",
        type=float,
        default=5.0,
        help="Threshold between medium and high label density. Default: 5.0.",
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Optional subset of cities for debugging. Default: all cities.",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N patches per city. Use 0 to disable. Default: 250.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output CSV/JSON/Markdown files.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root

    output_dir: Path = args.output_dir or (
        instance_root / "metadata" / "instance_C_patches"
    )

    input_csv: Path = args.input_csv or (
        output_dir
        / f"patch_tiling_index_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    )

    output_csv: Path = (
        output_dir
        / f"patch_metadata_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    )

    output_json: Path = (
        output_dir
        / f"patch_metadata_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.json"
    )

    output_md: Path = (
        output_dir
        / f"patch_metadata_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.md"
    )

    log("STEP", "Computing instance C patch metadata.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Input CSV:     {path_to_str(input_csv)}")
    log("INFO", f"Output CSV:    {path_to_str(output_csv)}")
    log("INFO", f"Output JSON:   {path_to_str(output_json)}")
    log("INFO", f"Output MD:     {path_to_str(output_md)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    rows = read_csv_rows(input_csv)
    validate_required_columns(rows)

    log("INFO", f"Loaded patch index rows: {len(rows)}")

    if args.cities:
        requested = set(args.cities)

        before_count = len(rows)
        rows = [row for row in rows if row["city"] in requested]
        after_count = len(rows)

        found_cities = set(row["city"] for row in rows)
        missing = sorted(requested - found_cities)

        if missing:
            fail(
                "Requested cities were not found in the patch index:\n"
                + "\n".join(f"  - {city}" for city in missing)
            )

        log(
            "WARN",
            f"Running on city subset. Rows before filter: {before_count}, after filter: {after_count}",
        )

    grouped = group_rows_by_city(rows)

    log("INFO", f"Cities to process: {len(grouped)}")
    log("INFO", f"S2 all-zero-as-nodata: {args.s2_all_zero_as_nodata}")
    log("INFO", f"S1 SNAP-GRD all-zero-as-nodata: {args.s1_all_zero_as_nodata}")
    log("INFO", f"S1 RTC all-zero-as-nodata: {args.s1_rtc_all_zero_as_nodata}")
    log("INFO", f"NaN/Inf-as-nodata: {args.nan_as_nodata}")

    all_metadata_rows: List[Dict[str, object]] = []

    for city, city_rows in grouped.items():
        log("STEP", f"Processing city: {city} ({len(city_rows)} patches)")

        city_metadata_rows = process_city_rows(
            city=city,
            city_rows=city_rows,
            s2_all_zero_as_nodata=args.s2_all_zero_as_nodata,
            s1_all_zero_as_nodata=args.s1_all_zero_as_nodata,
            s1_rtc_all_zero_as_nodata=args.s1_rtc_all_zero_as_nodata,
            nan_as_nodata=args.nan_as_nodata,
            low_density_threshold_percent=args.low_density_threshold_percent,
            high_density_threshold_percent=args.high_density_threshold_percent,
            progress_every=args.progress_every,
        )

        all_metadata_rows.extend(city_metadata_rows)

        city_positive = sum(int(row["patch_label_binary"]) for row in city_metadata_rows)
        min_s2_valid = min(float(row["s2_valid_percent"]) for row in city_metadata_rows)
        min_s1_valid = min(float(row["s1_snap_grd_valid_percent"]) for row in city_metadata_rows)
        max_label_positive = max(float(row["label_positive_percent"]) for row in city_metadata_rows)

        log(
            "OK",
            f"{city}: patches={len(city_metadata_rows)}, "
            f"positive={city_positive}, "
            f"max_label_positive={max_label_positive:.6f}%, "
            f"min_s2_valid={min_s2_valid:.6f}%, "
            f"min_s1_valid={min_s1_valid:.6f}%",
        )

    if not all_metadata_rows:
        fail("No metadata rows were generated.")

    log("STEP", "Building summary.")

    summary = build_summary(
        metadata_rows=all_metadata_rows,
        instance_root=instance_root,
        input_csv=input_csv,
        output_csv=output_csv,
        output_json=output_json,
        output_md=output_md,
        args=args,
    )

    log("STEP", "Writing outputs.")

    write_csv(output_csv, all_metadata_rows, overwrite=args.overwrite)
    write_json(output_json, summary, overwrite=args.overwrite)
    write_markdown(output_md, summary, overwrite=args.overwrite)

    log("OK", f"Wrote CSV:      {path_to_str(output_csv)}")
    log("OK", f"Wrote JSON:     {path_to_str(output_json)}")
    log("OK", f"Wrote Markdown: {path_to_str(output_md)}")

    log("STEP", "Final summary.")
    log("OK", f"Total patches: {summary['total_patches']}")
    log("OK", f"Positive patches: {summary['positive_patches']}")
    log("OK", f"Positive patch percent: {summary['positive_patch_percent']:.6f}%")
    log("OK", f"Empty patches: {summary['empty_patches']}")
    log("OK", f"Empty patch percent: {summary['empty_patch_percent']:.6f}%")
    log("OK", f"Label non-binary patches: {summary['label_non_binary_patches']}")
    log("OK", f"Patches with S2 valid percent < 100: {summary['patches_s2_valid_lt_100']}")
    log("OK", f"Patches with S1 SNAP-GRD valid percent < 100: {summary['patches_s1_snap_grd_valid_lt_100']}")
    log("OK", f"Patches with S1 RTC available: {summary['patches_with_s1_rtc']}")

    log("INFO", "Label density bins:")
    for key, value in summary["label_density_bin_counts"].items():
        log("INFO", f"  {key}: {value}")


if __name__ == "__main__":
    main()