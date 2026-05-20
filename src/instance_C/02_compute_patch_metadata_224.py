#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_compute_patch_metadata_224.py

Compute patch-level metadata for Instance C.

This script reads the patch tiling index created by:

    src/instance_C/01_build_patch_tiling_index_224.py

Input:

    <instance-root>/metadata/instance_C_patches/
        patch_tiling_index_ps224_st112_cover.csv

Outputs:

    <instance-root>/metadata/instance_C_patches/
        patch_metadata_ps224_st112_cover.csv
        patch_metadata_ps224_st112_cover.json
        patch_metadata_ps224_st112_cover.md

Important modality contract:

    S2:
        12 bands

    S1 SNAP-GRD:
        3 bands
        VV, VH, VV_minus_VH

    S1 RTC:
        2 bands
        VV, VH

    Label:
        1 band
        binary favela mask

This version explicitly supports 2-band RTC and computes RTC patch metadata.

Example:

python src/instance_C/02_compute_patch_metadata_224.py `
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
import math
import re
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
        "[ERROR] rasterio is required.\n"
        "Install it with:\n"
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


def normalize_city(value: str) -> str:
    value = str(value).strip()
    value = value.replace("\\", "/").split("/")[-1]
    value = value.lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_int(value: object, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


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
        fail(f"No rows to write: {path_to_str(path)}")

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
    summary: Dict[str, object],
    city_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Instance C patch metadata")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Input CSV: `{summary['input_csv']}`")
    lines.append(f"- Output CSV: `{summary['outputs']['csv']}`")
    lines.append(f"- Patch size: `{summary['parameters']['patch_size']}`")
    lines.append(f"- Stride: `{summary['parameters']['stride']}`")
    lines.append(f"- Edge mode: `{summary['parameters']['edge_mode']}`")
    lines.append("")
    lines.append("## Global patch counts")
    lines.append("")
    lines.append(f"- Total patches: `{summary['total_patches']}`")
    lines.append(f"- Positive patches: `{summary['positive_patches']}`")
    lines.append(f"- Positive patch percent: `{summary['positive_patch_percent']}`")
    lines.append(f"- Empty patches: `{summary['empty_patches']}`")
    lines.append(f"- Empty patch percent: `{summary['empty_patch_percent']}`")
    lines.append(f"- Label non-binary patches: `{summary['label_non_binary_patches']}`")
    lines.append("")
    lines.append("## Validity checks")
    lines.append("")
    lines.append(f"- Patches with S2 valid percent < 100: `{summary['patches_s2_valid_percent_lt_100']}`")
    lines.append(f"- Patches with S1 SNAP-GRD valid percent < 100: `{summary['patches_s1_snap_valid_percent_lt_100']}`")
    lines.append(f"- Patches with S1 RTC valid percent < 100: `{summary['patches_s1_rtc_valid_percent_lt_100']}`")
    lines.append(f"- Patches with S1 RTC available: `{summary['patches_with_s1_rtc_available']}`")
    lines.append(f"- Patches with S1 RTC all-zero percent > 0: `{summary['patches_s1_rtc_zero_percent_gt_0']}`")
    lines.append("")
    lines.append("## Label density bins")
    lines.append("")
    lines.append("| bin | patches |")
    lines.append("|---|---:|")
    for key, count in summary["label_density_bins"].items():
        lines.append(f"| {key} | {count} |")

    lines.append("")
    lines.append("## City-level summary")
    lines.append("")
    lines.append(
        "| city | region | patches | positive | max label % | min S2 valid % | "
        "min SNAP valid % | min RTC valid % | max RTC zero % | RTC available |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in city_rows:
        lines.append(
            f"| {row['city']} | "
            f"{row['region']} | "
            f"{row['patches']} | "
            f"{row['positive_patches']} | "
            f"{row['max_label_positive_percent']} | "
            f"{row['min_s2_valid_percent']} | "
            f"{row['min_s1_snap_valid_percent']} | "
            f"{row['min_s1_rtc_valid_percent']} | "
            f"{row['max_s1_rtc_zero_percent']} | "
            f"{row['patches_with_s1_rtc_available']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- This metadata file is based on the corrected Instance C patch index.")
    lines.append("- S1 RTC is expected to have 2 bands: VV and VH.")
    lines.append("- S1 SNAP-GRD is expected to have 3 bands: VV, VH, and VV_minus_VH.")
    lines.append("- For the main CROMA RTC-vs-SNAP comparison, both SAR variants should be loaded as VV/VH only.")
    lines.append("- `s1_rtc_zero_percent` measures pixels where both RTC VV and VH are zero.")
    lines.append("- After RTC cleaning, all patches should normally have `s1_rtc_valid_percent = 100` and `s1_rtc_zero_percent = 0`.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------

def masked_data_and_mask(array: np.ma.MaskedArray) -> Tuple[np.ndarray, np.ndarray]:
    data = np.ma.getdata(array)
    mask = np.ma.getmaskarray(array)

    if mask.shape == ():
        mask = np.zeros(data.shape, dtype=bool)

    return data, mask


def valid_mask_multiband(
    arr: np.ma.MaskedArray,
    *,
    all_zero_as_nodata: bool,
    nan_inf_as_nodata: bool,
) -> np.ndarray:
    data, mask = masked_data_and_mask(arr)

    if data.ndim != 3:
        raise ValueError(f"Expected multiband array shape (B, H, W), got {data.shape}")

    valid = np.ones(data.shape[1:], dtype=bool)

    for band_idx in range(data.shape[0]):
        valid &= ~mask[band_idx]

        if nan_inf_as_nodata:
            valid &= np.isfinite(data[band_idx])

    if all_zero_as_nodata:
        all_zero = np.ones(data.shape[1:], dtype=bool)
        for band_idx in range(data.shape[0]):
            all_zero &= data[band_idx] == 0
        valid &= ~all_zero

    return valid


def valid_mask_singleband(
    arr: np.ma.MaskedArray,
    *,
    nan_inf_as_nodata: bool,
) -> np.ndarray:
    data, mask = masked_data_and_mask(arr)

    if data.ndim != 2:
        raise ValueError(f"Expected single-band array shape (H, W), got {data.shape}")

    valid = ~mask

    if nan_inf_as_nodata:
        valid &= np.isfinite(data)

    return valid


def all_zero_percent_two_band(
    arr: np.ma.MaskedArray,
    *,
    zero_epsilon: float,
) -> float:
    data, mask = masked_data_and_mask(arr)

    if data.ndim != 3 or data.shape[0] < 2:
        raise ValueError(f"Expected at least 2 bands, got shape {data.shape}")

    valid = (
        ~mask[0]
        & ~mask[1]
        & np.isfinite(data[0])
        & np.isfinite(data[1])
    )

    both_zero = (
        valid
        & (np.abs(data[0]) <= zero_epsilon)
        & (np.abs(data[1]) <= zero_epsilon)
    )

    total = both_zero.size
    return 100.0 * int(np.count_nonzero(both_zero)) / total if total else 0.0


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def round_float(value: float, digits: int = 8) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return round(float(value), digits)


def band_stats(
    arr: np.ma.MaskedArray,
    band_index_zero_based: int,
    common_valid: np.ndarray,
) -> Dict[str, object]:
    data, mask = masked_data_and_mask(arr)

    band = data[band_index_zero_based]
    band_valid = common_valid & ~mask[band_index_zero_based] & np.isfinite(band)

    values = band[band_valid]

    if values.size == 0:
        return {
            "min": "",
            "max": "",
            "mean": "",
            "std": "",
        }

    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def label_stats(
    label_arr: np.ma.MaskedArray,
    *,
    nan_inf_as_nodata: bool,
) -> Dict[str, object]:
    data, mask = masked_data_and_mask(label_arr)
    valid = valid_mask_singleband(label_arr, nan_inf_as_nodata=nan_inf_as_nodata)

    total_pixels = int(data.size)
    valid_pixels = int(np.count_nonzero(valid))

    valid_values = data[valid]

    positive = valid & (data > 0)
    positive_pixels = int(np.count_nonzero(positive))
    positive_percent = percent(positive_pixels, total_pixels)

    if valid_values.size == 0:
        non_binary = True
        unique_values = ""
    else:
        unique = np.unique(valid_values)
        non_binary = bool(np.any(~np.isin(unique, [0, 1])))
        unique_values = ";".join(str(int(x)) if float(x).is_integer() else str(float(x)) for x in unique[:20])

    return {
        "label_total_pixels": total_pixels,
        "label_valid_pixels": valid_pixels,
        "label_positive_pixels": positive_pixels,
        "label_positive_percent": round_float(positive_percent),
        "patch_label_binary": 1 if positive_pixels > 0 else 0,
        "label_non_binary": bool(non_binary),
        "label_unique_values_sample": unique_values,
    }


def density_bin(label_positive_percent: float) -> str:
    value = float(label_positive_percent)

    if value <= 0.0:
        return "empty"
    if value < 1.0:
        return "low"
    if value < 10.0:
        return "medium"
    return "high"


# ---------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------

def group_rows_by_city(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[normalize_city(row["city"])].append(row)

    return dict(sorted(grouped.items()))


# ---------------------------------------------------------------------
# Per-patch processing
# ---------------------------------------------------------------------

def process_patch(
    row: Dict[str, str],
    *,
    s2,
    s1_snap,
    s1_rtc,
    label,
    args: argparse.Namespace,
) -> Dict[str, object]:
    city = normalize_city(row["city"])

    row_start = safe_int(row["row_start"])
    col_start = safe_int(row["col_start"])
    height = safe_int(row["height"])
    width = safe_int(row["width"])

    window = Window(
        col_off=col_start,
        row_off=row_start,
        width=width,
        height=height,
    )

    s2_arr = s2.read(window=window, masked=True)
    snap_arr = s1_snap.read(window=window, masked=True)
    rtc_arr = s1_rtc.read(window=window, masked=True)
    label_arr = label.read(1, window=window, masked=True)

    total_pixels = int(height * width)

    s2_valid = valid_mask_multiband(
        s2_arr,
        all_zero_as_nodata=bool(args.s2_all_zero_as_nodata),
        nan_inf_as_nodata=bool(args.nan_inf_as_nodata),
    )

    snap_valid = valid_mask_multiband(
        snap_arr,
        all_zero_as_nodata=bool(args.s1_snap_all_zero_as_nodata),
        nan_inf_as_nodata=bool(args.nan_inf_as_nodata),
    )

    rtc_valid = valid_mask_multiband(
        rtc_arr,
        all_zero_as_nodata=bool(args.s1_rtc_all_zero_as_nodata),
        nan_inf_as_nodata=bool(args.nan_inf_as_nodata),
    )

    s2_valid_pixels = int(np.count_nonzero(s2_valid))
    snap_valid_pixels = int(np.count_nonzero(snap_valid))
    rtc_valid_pixels = int(np.count_nonzero(rtc_valid))

    s2_valid_percent = percent(s2_valid_pixels, total_pixels)
    snap_valid_percent = percent(snap_valid_pixels, total_pixels)
    rtc_valid_percent = percent(rtc_valid_pixels, total_pixels)

    rtc_zero_percent = all_zero_percent_two_band(
        rtc_arr,
        zero_epsilon=float(args.zero_epsilon),
    )

    snap_zero_percent = all_zero_percent_two_band(
        snap_arr,
        zero_epsilon=float(args.zero_epsilon),
    )

    lab = label_stats(
        label_arr,
        nan_inf_as_nodata=bool(args.nan_inf_as_nodata),
    )

    snap_vv = band_stats(snap_arr, 0, snap_valid)
    snap_vh = band_stats(snap_arr, 1, snap_valid)
    snap_vvdiff = band_stats(snap_arr, 2, snap_valid) if snap_arr.shape[0] >= 3 else {"min": "", "max": "", "mean": "", "std": ""}

    rtc_vv = band_stats(rtc_arr, 0, rtc_valid)
    rtc_vh = band_stats(rtc_arr, 1, rtc_valid)

    out: Dict[str, object] = {
        "patch_id": row["patch_id"],
        "city": city,
        "region": row.get("region", ""),
        "row_start": row_start,
        "col_start": col_start,
        "height": height,
        "width": width,
        "patch_size": safe_int(row.get("patch_size", height)),
        "stride": safe_int(row.get("stride", 0)),
        "edge_mode": row.get("edge_mode", ""),
        "city_width": safe_int(row.get("city_width", 0)),
        "city_height": safe_int(row.get("city_height", 0)),

        "source_s2_path": row.get("source_s2_path", ""),
        "source_s1_path": row.get("source_s1_path", row.get("source_s1_snap_path", "")),
        "source_s1_snap_path": row.get("source_s1_snap_path", row.get("source_s1_path", "")),
        "source_s1_rtc_path": row.get("source_s1_rtc_path", ""),
        "source_label_path": row.get("source_label_path", ""),

        "s2_exists": row.get("s2_exists", "True"),
        "s1_exists": row.get("s1_exists", "True"),
        "s1_snap_exists": row.get("s1_snap_exists", "True"),
        "s1_rtc_exists": row.get("s1_rtc_exists", "True"),
        "label_exists": row.get("label_exists", "True"),

        "s2_band_count": safe_int(row.get("s2_band_count", s2.count)),
        "s1_band_count": safe_int(row.get("s1_band_count", s1_snap.count)),
        "s1_snap_band_count": safe_int(row.get("s1_snap_band_count", s1_snap.count)),
        "s1_rtc_band_count": safe_int(row.get("s1_rtc_band_count", s1_rtc.count)),
        "label_band_count": safe_int(row.get("label_band_count", label.count)),

        "s2_valid_pixels": s2_valid_pixels,
        "s2_valid_percent": round_float(s2_valid_percent),

        "s1_valid_pixels": snap_valid_pixels,
        "s1_valid_percent": round_float(snap_valid_percent),

        "s1_snap_valid_pixels": snap_valid_pixels,
        "s1_snap_valid_percent": round_float(snap_valid_percent),
        "s1_snap_zero_percent": round_float(snap_zero_percent),

        "s1_rtc_available": True,
        "s1_rtc_valid_pixels": rtc_valid_pixels,
        "s1_rtc_valid_percent": round_float(rtc_valid_percent),
        "s1_rtc_zero_percent": round_float(rtc_zero_percent),
    }

    out.update(lab)
    out["label_density_bin"] = density_bin(float(out["label_positive_percent"]))

    out.update(
        {
            "s1_snap_vv_min": snap_vv["min"],
            "s1_snap_vv_max": snap_vv["max"],
            "s1_snap_vv_mean": snap_vv["mean"],
            "s1_snap_vv_std": snap_vv["std"],
            "s1_snap_vh_min": snap_vh["min"],
            "s1_snap_vh_max": snap_vh["max"],
            "s1_snap_vh_mean": snap_vh["mean"],
            "s1_snap_vh_std": snap_vh["std"],
            "s1_snap_vvdiff_min": snap_vvdiff["min"],
            "s1_snap_vvdiff_max": snap_vvdiff["max"],
            "s1_snap_vvdiff_mean": snap_vvdiff["mean"],
            "s1_snap_vvdiff_std": snap_vvdiff["std"],

            "s1_rtc_vv_min": rtc_vv["min"],
            "s1_rtc_vv_max": rtc_vv["max"],
            "s1_rtc_vv_mean": rtc_vv["mean"],
            "s1_rtc_vv_std": rtc_vv["std"],
            "s1_rtc_vh_min": rtc_vh["min"],
            "s1_rtc_vh_max": rtc_vh["max"],
            "s1_rtc_vh_mean": rtc_vh["mean"],
            "s1_rtc_vh_std": rtc_vh["std"],
        }
    )

    return out


def validate_city_rasters(
    *,
    city: str,
    s2,
    s1_snap,
    s1_rtc,
    label,
) -> None:
    if s2.count != 12:
        raise ValueError(f"{city}: S2 band count = {s2.count}, expected 12")

    if s1_snap.count != 3:
        raise ValueError(f"{city}: S1 SNAP-GRD band count = {s1_snap.count}, expected 3")

    if s1_rtc.count != 2:
        raise ValueError(f"{city}: S1 RTC band count = {s1_rtc.count}, expected 2")

    if label.count != 1:
        raise ValueError(f"{city}: Label band count = {label.count}, expected 1")

    ref = (s2.width, s2.height, s2.crs, s2.transform)

    for name, src in [
        ("S1 SNAP-GRD", s1_snap),
        ("S1 RTC", s1_rtc),
        ("Label", label),
    ]:
        if src.width != ref[0] or src.height != ref[1]:
            raise ValueError(
                f"{city}: {name} shape mismatch: {src.width}x{src.height} != {ref[0]}x{ref[1]}"
            )

        if src.crs != ref[2]:
            raise ValueError(f"{city}: {name} CRS mismatch")

        if src.transform != ref[3]:
            raise ValueError(f"{city}: {name} transform mismatch")


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize_city(city: str, rows: List[Dict[str, object]]) -> Dict[str, object]:
    region = rows[0].get("region", "")

    label_percents = [safe_float(r["label_positive_percent"]) for r in rows]
    s2_valids = [safe_float(r["s2_valid_percent"]) for r in rows]
    snap_valids = [safe_float(r["s1_snap_valid_percent"]) for r in rows]
    rtc_valids = [safe_float(r["s1_rtc_valid_percent"]) for r in rows]
    rtc_zeros = [safe_float(r["s1_rtc_zero_percent"]) for r in rows]

    return {
        "city": city,
        "region": region,
        "patches": len(rows),
        "positive_patches": sum(1 for r in rows if safe_int(r["patch_label_binary"]) == 1),
        "max_label_positive_percent": round_float(max(label_percents) if label_percents else 0.0),
        "min_s2_valid_percent": round_float(min(s2_valids) if s2_valids else 0.0),
        "min_s1_snap_valid_percent": round_float(min(snap_valids) if snap_valids else 0.0),
        "min_s1_rtc_valid_percent": round_float(min(rtc_valids) if rtc_valids else 0.0),
        "max_s1_rtc_zero_percent": round_float(max(rtc_zeros) if rtc_zeros else 0.0),
        "patches_with_s1_rtc_available": sum(1 for r in rows if parse_bool(r["s1_rtc_available"])),
    }


def build_summary(
    *,
    instance_root: Path,
    input_csv: Path,
    output_csv: Path,
    output_json: Path,
    output_md: Path,
    metadata_rows: List[Dict[str, object]],
    city_summary_rows: List[Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    total = len(metadata_rows)
    positive = sum(1 for r in metadata_rows if safe_int(r["patch_label_binary"]) == 1)
    empty = total - positive

    density_counter = Counter(str(r["label_density_bin"]) for r in metadata_rows)

    ordered_bins = {
        "empty": density_counter.get("empty", 0),
        "low": density_counter.get("low", 0),
        "medium": density_counter.get("medium", 0),
        "high": density_counter.get("high", 0),
    }

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "input_csv": path_to_str(input_csv),
        "total_patches": total,
        "positive_patches": positive,
        "positive_patch_percent": round_float(percent(positive, total)),
        "empty_patches": empty,
        "empty_patch_percent": round_float(percent(empty, total)),
        "label_non_binary_patches": sum(1 for r in metadata_rows if parse_bool(r["label_non_binary"])),
        "patches_s2_valid_percent_lt_100": sum(1 for r in metadata_rows if safe_float(r["s2_valid_percent"]) < 100.0),
        "patches_s1_snap_valid_percent_lt_100": sum(1 for r in metadata_rows if safe_float(r["s1_snap_valid_percent"]) < 100.0),
        "patches_s1_rtc_valid_percent_lt_100": sum(1 for r in metadata_rows if safe_float(r["s1_rtc_valid_percent"]) < 100.0),
        "patches_with_s1_rtc_available": sum(1 for r in metadata_rows if parse_bool(r["s1_rtc_available"])),
        "patches_s1_rtc_zero_percent_gt_0": sum(1 for r in metadata_rows if safe_float(r["s1_rtc_zero_percent"]) > 0.0),
        "label_density_bins": ordered_bins,
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "s2_all_zero_as_nodata": bool(args.s2_all_zero_as_nodata),
            "s1_snap_all_zero_as_nodata": bool(args.s1_snap_all_zero_as_nodata),
            "s1_rtc_all_zero_as_nodata": bool(args.s1_rtc_all_zero_as_nodata),
            "nan_inf_as_nodata": bool(args.nan_inf_as_nodata),
            "zero_epsilon": args.zero_epsilon,
        },
        "outputs": {
            "csv": path_to_str(output_csv),
            "json": path_to_str(output_json),
            "markdown": path_to_str(output_md),
        },
        "city_rows": city_summary_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Instance C patch metadata with 2-band RTC support."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Patch tiling index CSV. Default: metadata/instance_C_patches/patch_tiling_index_ps<patch-size>_st<stride>_<edge-mode>.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <instance-root>/metadata/instance_C_patches.",
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
        help="Patch stride. Default: 112.",
    )

    parser.add_argument(
        "--edge-mode",
        choices=["cover", "drop"],
        default="cover",
        help="Edge mode. Default: cover.",
    )

    parser.add_argument(
        "--s2-all-zero-as-nodata",
        action="store_true",
        help="Treat all-zero S2 pixels as nodata. Default: False.",
    )

    parser.add_argument(
        "--s1-snap-all-zero-as-nodata",
        action="store_true",
        help="Treat all-zero S1 SNAP-GRD pixels as nodata. Default: False.",
    )

    parser.add_argument(
        "--s1-rtc-all-zero-as-nodata",
        action="store_true",
        help="Treat all-zero S1 RTC pixels as nodata. Default: False.",
    )

    parser.add_argument(
        "--nan-inf-as-nodata",
        action="store_true",
        help="Treat NaN/Inf as nodata. Default: False.",
    )

    parser.add_argument(
        "--zero-epsilon",
        type=float,
        default=1e-6,
        help="Tolerance for all-zero VV/VH detection. Default: 1e-6.",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N patches per city. Default: 250.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root
    output_dir: Path = args.output_dir or (instance_root / "metadata" / "instance_C_patches")

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    input_csv: Path = args.input_csv or (
        output_dir / f"patch_tiling_index_{stem}.csv"
    )

    output_csv = output_dir / f"patch_metadata_{stem}.csv"
    output_json = output_dir / f"patch_metadata_{stem}.json"
    output_md = output_dir / f"patch_metadata_{stem}.md"

    log("STEP", "Computing instance C patch metadata.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Input CSV:     {path_to_str(input_csv)}")
    log("INFO", f"Output CSV:    {path_to_str(output_csv)}")
    log("INFO", f"Output JSON:   {path_to_str(output_json)}")
    log("INFO", f"Output MD:     {path_to_str(output_md)}")
    log("INFO", f"S2 all-zero-as-nodata: {args.s2_all_zero_as_nodata}")
    log("INFO", f"S1 SNAP-GRD all-zero-as-nodata: {args.s1_snap_all_zero_as_nodata}")
    log("INFO", f"S1 RTC all-zero-as-nodata: {args.s1_rtc_all_zero_as_nodata}")
    log("INFO", f"NaN/Inf-as-nodata: {args.nan_inf_as_nodata}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    index_rows = read_csv_rows(input_csv)

    log("INFO", f"Loaded patch index rows: {len(index_rows)}")

    grouped = group_rows_by_city(index_rows)

    log("INFO", f"Cities to process: {len(grouped)}")

    all_metadata_rows: List[Dict[str, object]] = []
    city_summary_rows: List[Dict[str, object]] = []

    for city, city_rows in grouped.items():
        log("STEP", f"Processing city: {city} ({len(city_rows)} patches)")

        first = city_rows[0]

        s2_path = Path(first["source_s2_path"])
        snap_path = Path(first.get("source_s1_snap_path") or first.get("source_s1_path"))
        rtc_path = Path(first["source_s1_rtc_path"])
        label_path = Path(first["source_label_path"])

        for label_name, path in [
            ("S2", s2_path),
            ("S1 SNAP-GRD", snap_path),
            ("S1 RTC", rtc_path),
            ("Label", label_path),
        ]:
            if not path.exists():
                fail(f"{city}: missing {label_name} raster: {path_to_str(path)}")

        city_metadata: List[Dict[str, object]] = []

        with rasterio.open(s2_path) as s2, \
             rasterio.open(snap_path) as s1_snap, \
             rasterio.open(rtc_path) as s1_rtc, \
             rasterio.open(label_path) as label:

            validate_city_rasters(
                city=city,
                s2=s2,
                s1_snap=s1_snap,
                s1_rtc=s1_rtc,
                label=label,
            )

            for idx, row in enumerate(city_rows, start=1):
                metadata_row = process_patch(
                    row,
                    s2=s2,
                    s1_snap=s1_snap,
                    s1_rtc=s1_rtc,
                    label=label,
                    args=args,
                )

                city_metadata.append(metadata_row)

                if args.progress_every > 0 and idx % int(args.progress_every) == 0:
                    log("INFO", f"{city}: processed {idx}/{len(city_rows)} patches")

        all_metadata_rows.extend(city_metadata)

        city_summary = summarize_city(city, city_metadata)
        city_summary_rows.append(city_summary)

        log(
            "OK",
            f"{city}: patches={city_summary['patches']}, "
            f"positive={city_summary['positive_patches']}, "
            f"max_label_positive={city_summary['max_label_positive_percent']:.6f}%, "
            f"min_s2_valid={city_summary['min_s2_valid_percent']:.6f}%, "
            f"min_s1_snap_valid={city_summary['min_s1_snap_valid_percent']:.6f}%, "
            f"min_s1_rtc_valid={city_summary['min_s1_rtc_valid_percent']:.6f}%, "
            f"max_s1_rtc_zero={city_summary['max_s1_rtc_zero_percent']:.6f}%",
        )

    log("STEP", "Building summary.")

    summary = build_summary(
        instance_root=instance_root,
        input_csv=input_csv,
        output_csv=output_csv,
        output_json=output_json,
        output_md=output_md,
        metadata_rows=all_metadata_rows,
        city_summary_rows=city_summary_rows,
        args=args,
    )

    log("STEP", "Writing outputs.")

    write_csv(output_csv, all_metadata_rows, overwrite=bool(args.overwrite))
    write_json(output_json, summary, overwrite=bool(args.overwrite))
    write_markdown(output_md, summary, city_summary_rows, overwrite=bool(args.overwrite))

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
    log("OK", f"Patches with S2 valid percent < 100: {summary['patches_s2_valid_percent_lt_100']}")
    log("OK", f"Patches with S1 SNAP-GRD valid percent < 100: {summary['patches_s1_snap_valid_percent_lt_100']}")
    log("OK", f"Patches with S1 RTC valid percent < 100: {summary['patches_s1_rtc_valid_percent_lt_100']}")
    log("OK", f"Patches with S1 RTC available: {summary['patches_with_s1_rtc_available']}")
    log("OK", f"Patches with S1 RTC all-zero percent > 0: {summary['patches_s1_rtc_zero_percent_gt_0']}")

    log("INFO", "Label density bins:")
    for key, count in summary["label_density_bins"].items():
        log("INFO", f"  {key}: {count}")


if __name__ == "__main__":
    main()