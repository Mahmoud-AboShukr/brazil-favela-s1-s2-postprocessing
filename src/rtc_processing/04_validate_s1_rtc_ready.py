#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_validate_s1_rtc_ready.py

Validate finalized S1 RTC rasters inside Instance C.

This script validates the outputs created by:

    src/rtc_processing/03_finalize_s1_rtc_ready.py

Expected RTC outputs:

    <instance-root>/s1_rtc_ready/<city>/<city>_s1_rtc_vv_vh_10m_aligned.tif

Expected convention:

    band 1 = VV
    band 2 = VH

This script checks:

    1. RTC exists for each city.
    2. RTC has exactly 2 bands.
    3. RTC shape/CRS/transform exactly match the S2 reference raster.
    4. RTC shape/CRS/transform match the label raster.
    5. RTC contains finite values.
    6. RTC all-zero VV/VH pixels are quantified as likely filled/no-coverage pixels.
    7. RTC all-zero pixels are summarized at city level.
    8. RTC all-zero pixels are summarized at 224x224 patch level.
    9. Overlap between zero-filled RTC pixels and favela label pixels is quantified.

Important:

The finalization script filled invalid/uncovered pixels with 0.0 by default.
Therefore, pixels where both VV and VH are exactly zero are treated here as
"likely filled / likely no-coverage" pixels.

This is a diagnostic proxy. It is strong because both VV and VH being exactly
zero after float reprojection is unlikely for real dB-like SAR signal, but it
should still be interpreted as a QC indicator rather than absolute truth.

Outputs:

    <instance-root>/metadata/rtc_processing/s1_rtc_ready_validation_city.csv
    <instance-root>/metadata/rtc_processing/s1_rtc_ready_validation_patches_ps224_st112_cover.csv
    <instance-root>/metadata/rtc_processing/s1_rtc_ready_validation.json
    <instance-root>/metadata/rtc_processing/s1_rtc_ready_validation.md

Example:

python src/rtc_processing/04_validate_s1_rtc_ready.py `
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_city(value: str) -> str:
    value = str(value).strip()
    value = value.replace("\\", "/").split("/")[-1]
    value = value.lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def bool_to_text(value: bool) -> str:
    return "True" if value else "False"


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
        fail(f"No rows to write for CSV: {path_to_str(path)}")

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

    lines.append("# S1 RTC ready validation")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- RTC root: `{summary['rtc_root']}`")
    lines.append(f"- Patch metadata CSV: `{summary['patch_metadata_csv']}`")
    lines.append(f"- Finalization CSV: `{summary['finalization_csv']}`")
    lines.append(f"- Cities validated: `{summary['n_cities_validated']}`")
    lines.append(f"- Cities with grid OK: `{summary['n_cities_grid_ok']}`")
    lines.append(f"- Cities with band count OK: `{summary['n_cities_band_count_ok']}`")
    lines.append(f"- Cities with finite percent < 100: `{summary['n_cities_finite_lt_100']}`")
    lines.append(f"- Cities with zero-filled percent > threshold: `{summary['n_cities_zero_gt_threshold']}`")
    lines.append(f"- Cities failed validation: `{summary['n_cities_failed']}`")
    lines.append("")

    lines.append("## Parameters")
    lines.append("")
    params = summary["parameters"]
    lines.append(f"- Zero epsilon: `{params['zero_epsilon']}`")
    lines.append(f"- Zero warning threshold percent: `{params['zero_warning_threshold_percent']}`")
    lines.append(f"- Patch size: `{params['patch_size']}`")
    lines.append(f"- Stride: `{params['stride']}`")
    lines.append(f"- Edge mode: `{params['edge_mode']}`")
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    outputs = summary["outputs"]
    lines.append(f"- City CSV: `{outputs['city_csv']}`")
    lines.append(f"- Patch CSV: `{outputs['patch_csv']}`")
    lines.append(f"- JSON: `{outputs['json']}`")
    lines.append(f"- Markdown: `{outputs['markdown']}`")
    lines.append("")

    lines.append("## City-level validation")
    lines.append("")
    lines.append(
        "| city | status | route | grid OK | bands OK | finite % | all-zero % | "
        "label pixels affected % | patches any zero | positive patches any zero | notes |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

    for row in city_rows:
        lines.append(
            f"| {row['city']} | "
            f"{row['status']} | "
            f"{row['finalization_route']} | "
            f"{row['grid_matches_s2']} | "
            f"{row['band_count_ok']} | "
            f"{row['rtc_finite_percent']} | "
            f"{row['rtc_all_zero_percent']} | "
            f"{row['label_positive_pixels_affected_by_zero_percent']} | "
            f"{row['patches_any_zero']} | "
            f"{row['positive_patches_any_zero']} | "
            f"{row['notes']} |"
        )

    lines.append("")
    lines.append("## Cities with highest likely zero-filled RTC coverage")
    lines.append("")
    lines.append("| city | all-zero % | finalizer invalid-before-fill % | route |")
    lines.append("|---|---:|---:|---|")

    sorted_by_zero = sorted(
        city_rows,
        key=lambda r: safe_float(r.get("rtc_all_zero_percent", 0.0)),
        reverse=True,
    )

    for row in sorted_by_zero[:10]:
        lines.append(
            f"| {row['city']} | "
            f"{row['rtc_all_zero_percent']} | "
            f"{row['finalizer_max_invalid_percent_before_fill']} | "
            f"{row['finalization_route']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `grid OK` means the RTC raster has the same CRS, transform, width, and height as the S2 reference.")
    lines.append("- `bands OK` means the RTC raster has exactly 2 bands: VV and VH.")
    lines.append("- `all-zero %` is the percentage of pixels where both VV and VH are zero within the finalized RTC raster.")
    lines.append("- Since the finalizer filled invalid/uncovered pixels with 0.0, all-zero VV/VH pixels are treated as likely filled/no-coverage pixels.")
    lines.append("- A high all-zero percentage may indicate that the RTC source footprint did not fully cover the S2 reference extent.")
    lines.append("- Patch-level zero statistics should be used later to decide whether to repair, exclude, or flag affected patches.")
    lines.append("- The route difference itself is not a problem if the final grid validation passes; the main concern is zero-filled coverage.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Raster discovery
# ---------------------------------------------------------------------

EXCLUDE_NAME_PARTS = (
    "valid_mask",
    "fill_level",
    "fill_source",
    "nodata",
    "qa",
    "mask_before",
    "mask_after",
    "summary",
)


def is_excluded_raster(path: Path) -> bool:
    lower = path.name.lower()
    return any(part in lower for part in EXCLUDE_NAME_PARTS)


def candidate_tifs(folder: Path, patterns: Sequence[str]) -> List[Path]:
    matches: List[Path] = []

    for pattern in patterns:
        for path in folder.glob(pattern):
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
                if not is_excluded_raster(path):
                    matches.append(path)

    return sorted(set(matches))


def find_s2_reference(instance_root: Path, city: str) -> Path:
    city = normalize_city(city)
    city_dir = instance_root / "s2_filled" / city

    if not city_dir.exists():
        fail(f"Missing S2 folder for {city}: {path_to_str(city_dir)}")

    patterns = [
        f"{city}_s2_12bands_reflectance_10m.tif",
        f"{city}_s2_12bands_reflectance_10m.tiff",
        f"{city}_s2_12bands_reflectance_10m_filled.tif",
        f"{city}_s2_filled_12bands_reflectance_10m.tif",
        f"{city}*s2*12*reflectance*10m*.tif",
        f"{city}*s2*filled*.tif",
        "*s2*12*reflectance*10m*.tif",
        "*s2*filled*.tif",
        "*.tif",
        "*.tiff",
    ]

    for pattern in patterns:
        matches = candidate_tifs(city_dir, [pattern])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            formatted = "\n".join(f"  - {path_to_str(p)}" for p in matches[:20])
            fail(f"Ambiguous S2 reference for {city}:\n{formatted}")

    fail(f"Could not find S2 reference for {city} in {path_to_str(city_dir)}")


def find_label_reference(instance_root: Path, city: str) -> Path:
    city = normalize_city(city)
    city_dir = instance_root / "labels" / city

    if not city_dir.exists():
        fail(f"Missing label folder for {city}: {path_to_str(city_dir)}")

    patterns = [
        f"{city}_label_final.tif",
        f"{city}_label_final.tiff",
        f"{city}*label_final*.tif",
        f"{city}*label*.tif",
        "*label_final*.tif",
        "*label*.tif",
        "*.tif",
        "*.tiff",
    ]

    for pattern in patterns:
        matches = candidate_tifs(city_dir, [pattern])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            formatted = "\n".join(f"  - {path_to_str(p)}" for p in matches[:20])
            fail(f"Ambiguous label reference for {city}:\n{formatted}")

    fail(f"Could not find label reference for {city} in {path_to_str(city_dir)}")


def find_rtc_ready(instance_root: Path, city: str) -> Path:
    city = normalize_city(city)
    city_dir = instance_root / "s1_rtc_ready" / city

    if not city_dir.exists():
        fail(f"Missing RTC-ready folder for {city}: {path_to_str(city_dir)}")

    patterns = [
        f"{city}_s1_rtc_vv_vh_10m_aligned.tif",
        f"{city}_s1_rtc_vv_vh_10m_aligned.tiff",
        f"{city}*s1_rtc*vv*vh*10m*aligned*.tif",
        f"{city}*rtc*vv*vh*.tif",
        "*s1_rtc*vv*vh*10m*aligned*.tif",
        "*rtc*vv*vh*.tif",
        "*.tif",
        "*.tiff",
    ]

    for pattern in patterns:
        matches = candidate_tifs(city_dir, [pattern])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            formatted = "\n".join(f"  - {path_to_str(p)}" for p in matches[:20])
            fail(f"Ambiguous RTC-ready raster for {city}:\n{formatted}")

    fail(f"Could not find RTC-ready raster for {city} in {path_to_str(city_dir)}")


# ---------------------------------------------------------------------
# Grid / array helpers
# ---------------------------------------------------------------------

def affine_six(transform) -> Tuple[float, float, float, float, float, float]:
    return (
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    )


def transforms_equal(a, b, tolerance: float) -> bool:
    aa = affine_six(a)
    bb = affine_six(b)

    return all(abs(x - y) <= tolerance for x, y in zip(aa, bb))


def masked_data_and_mask(array: np.ma.MaskedArray) -> Tuple[np.ndarray, np.ndarray]:
    data = np.ma.getdata(array)
    mask = np.ma.getmaskarray(array)

    if mask.shape == ():
        mask = np.zeros(data.shape, dtype=bool)

    return data, mask


def compute_rtc_zero_mask(
    rtc_window: np.ma.MaskedArray,
    zero_epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
        finite_both: H x W bool
        all_zero_both: H x W bool

    rtc_window shape should be 2 x H x W.
    """

    data, mask = masked_data_and_mask(rtc_window)

    if data.ndim != 3 or data.shape[0] != 2:
        raise ValueError(f"Expected RTC window shape (2, H, W), got {data.shape}")

    finite_both = np.isfinite(data[0]) & np.isfinite(data[1])
    unmasked_both = ~mask[0] & ~mask[1]

    finite_both &= unmasked_both

    all_zero_both = (
        finite_both
        & (np.abs(data[0]) <= zero_epsilon)
        & (np.abs(data[1]) <= zero_epsilon)
    )

    return finite_both, all_zero_both


def compute_label_positive_mask(label_window: np.ma.MaskedArray) -> np.ndarray:
    data, mask = masked_data_and_mask(label_window)

    if data.ndim != 2:
        raise ValueError(f"Expected label window shape (H, W), got {data.shape}")

    valid = ~mask

    if np.issubdtype(data.dtype, np.floating):
        valid &= np.isfinite(data)

    return valid & (data > 0)


# ---------------------------------------------------------------------
# Finalization metadata
# ---------------------------------------------------------------------

def read_finalization_by_city(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}

    rows = read_csv_rows(path)

    return {
        normalize_city(row["city"]): row
        for row in rows
    }


# ---------------------------------------------------------------------
# Patch metadata
# ---------------------------------------------------------------------

def default_patch_metadata_path(
    instance_root: Path,
    patch_size: int,
    stride: int,
    edge_mode: str,
) -> Path:
    return (
        instance_root
        / "metadata"
        / "instance_C_patches"
        / f"patch_metadata_ps{patch_size}_st{stride}_{edge_mode}.csv"
    )


def group_patch_rows_by_city(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[normalize_city(row["city"])].append(row)

    return dict(sorted(grouped.items()))


# ---------------------------------------------------------------------
# City-level stats
# ---------------------------------------------------------------------

def validate_city_grid(
    city: str,
    rtc_path: Path,
    s2_path: Path,
    label_path: Path,
    transform_tolerance: float,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "rtc_exists": rtc_path.exists(),
        "s2_exists": s2_path.exists(),
        "label_exists": label_path.exists(),
        "rtc_band_count": "",
        "rtc_dtypes": "",
        "rtc_descriptions": "",
        "rtc_width": "",
        "rtc_height": "",
        "rtc_crs": "",
        "s2_width": "",
        "s2_height": "",
        "s2_crs": "",
        "label_width": "",
        "label_height": "",
        "label_crs": "",
        "band_count_ok": False,
        "shape_matches_s2": False,
        "crs_matches_s2": False,
        "transform_matches_s2": False,
        "label_grid_matches_s2": False,
        "grid_matches_s2": False,
        "open_error": "",
    }

    try:
        with rasterio.open(rtc_path) as rtc, rasterio.open(s2_path) as s2, rasterio.open(label_path) as label:
            result["rtc_band_count"] = rtc.count
            result["rtc_dtypes"] = ";".join(str(x) for x in rtc.dtypes)
            result["rtc_descriptions"] = ";".join("" if d is None else str(d) for d in rtc.descriptions)
            result["rtc_width"] = rtc.width
            result["rtc_height"] = rtc.height
            result["rtc_crs"] = str(rtc.crs)

            result["s2_width"] = s2.width
            result["s2_height"] = s2.height
            result["s2_crs"] = str(s2.crs)

            result["label_width"] = label.width
            result["label_height"] = label.height
            result["label_crs"] = str(label.crs)

            band_count_ok = rtc.count == 2
            shape_matches_s2 = rtc.width == s2.width and rtc.height == s2.height
            crs_matches_s2 = rtc.crs == s2.crs
            transform_matches_s2 = transforms_equal(rtc.transform, s2.transform, transform_tolerance)

            label_grid_matches_s2 = (
                label.width == s2.width
                and label.height == s2.height
                and label.crs == s2.crs
                and transforms_equal(label.transform, s2.transform, transform_tolerance)
            )

            grid_matches_s2 = (
                band_count_ok
                and shape_matches_s2
                and crs_matches_s2
                and transform_matches_s2
            )

            result["band_count_ok"] = band_count_ok
            result["shape_matches_s2"] = shape_matches_s2
            result["crs_matches_s2"] = crs_matches_s2
            result["transform_matches_s2"] = transform_matches_s2
            result["label_grid_matches_s2"] = label_grid_matches_s2
            result["grid_matches_s2"] = grid_matches_s2

    except Exception as exc:
        result["open_error"] = repr(exc)

    return result


def compute_city_rtc_stats(
    rtc_path: Path,
    label_path: Path,
    zero_epsilon: float,
) -> Dict[str, object]:
    total_pixels = 0
    finite_pixels = 0
    nonfinite_pixels = 0
    all_zero_pixels = 0

    label_positive_pixels = 0
    label_positive_pixels_affected_by_zero = 0

    vv_min = math.inf
    vv_max = -math.inf
    vv_sum = 0.0
    vv_count = 0

    vh_min = math.inf
    vh_max = -math.inf
    vh_sum = 0.0
    vh_count = 0

    with rasterio.open(rtc_path) as rtc, rasterio.open(label_path) as label:
        for _, window in rtc.block_windows(1):
            rtc_window = rtc.read([1, 2], window=window, masked=True)
            label_window = label.read(1, window=window, masked=True)

            finite_both, zero_both = compute_rtc_zero_mask(rtc_window, zero_epsilon)
            label_positive = compute_label_positive_mask(label_window)

            h, w = finite_both.shape
            n = h * w

            total_pixels += int(n)
            finite_pixels += int(np.count_nonzero(finite_both))
            nonfinite_pixels += int(n - np.count_nonzero(finite_both))
            all_zero_pixels += int(np.count_nonzero(zero_both))

            label_positive_pixels += int(np.count_nonzero(label_positive))
            label_positive_pixels_affected_by_zero += int(np.count_nonzero(label_positive & zero_both))

            data, mask = masked_data_and_mask(rtc_window)

            vv_valid = finite_both & ~zero_both
            vh_valid = finite_both & ~zero_both

            vv_values = data[0][vv_valid]
            vh_values = data[1][vh_valid]

            if vv_values.size > 0:
                vv_min = min(vv_min, float(np.min(vv_values)))
                vv_max = max(vv_max, float(np.max(vv_values)))
                vv_sum += float(np.sum(vv_values))
                vv_count += int(vv_values.size)

            if vh_values.size > 0:
                vh_min = min(vh_min, float(np.min(vh_values)))
                vh_max = max(vh_max, float(np.max(vh_values)))
                vh_sum += float(np.sum(vh_values))
                vh_count += int(vh_values.size)

    finite_percent = 100.0 * finite_pixels / total_pixels if total_pixels else 0.0
    nonfinite_percent = 100.0 * nonfinite_pixels / total_pixels if total_pixels else 0.0
    zero_percent = 100.0 * all_zero_pixels / total_pixels if total_pixels else 0.0

    label_affected_percent = (
        100.0 * label_positive_pixels_affected_by_zero / label_positive_pixels
        if label_positive_pixels else 0.0
    )

    return {
        "rtc_total_pixels": total_pixels,
        "rtc_finite_pixels": finite_pixels,
        "rtc_nonfinite_pixels": nonfinite_pixels,
        "rtc_finite_percent": round(finite_percent, 8),
        "rtc_nonfinite_percent": round(nonfinite_percent, 8),
        "rtc_all_zero_pixels": all_zero_pixels,
        "rtc_all_zero_percent": round(zero_percent, 8),
        "label_positive_pixels": label_positive_pixels,
        "label_positive_pixels_affected_by_zero": label_positive_pixels_affected_by_zero,
        "label_positive_pixels_affected_by_zero_percent": round(label_affected_percent, 8),
        "vv_min_nonzero": "" if vv_count == 0 else float(vv_min),
        "vv_max_nonzero": "" if vv_count == 0 else float(vv_max),
        "vv_mean_nonzero": "" if vv_count == 0 else float(vv_sum / vv_count),
        "vh_min_nonzero": "" if vh_count == 0 else float(vh_min),
        "vh_max_nonzero": "" if vh_count == 0 else float(vh_max),
        "vh_mean_nonzero": "" if vh_count == 0 else float(vh_sum / vh_count),
    }


# ---------------------------------------------------------------------
# Patch-level stats
# ---------------------------------------------------------------------

def compute_patch_stats_for_city(
    city: str,
    patch_rows: List[Dict[str, str]],
    rtc_path: Path,
    label_path: Path,
    zero_epsilon: float,
    patch_zero_thresholds: Sequence[float],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    patch_outputs: List[Dict[str, object]] = []

    threshold_counts = {
        f"patches_zero_ge_{str(th).replace('.', 'p')}pct": 0
        for th in patch_zero_thresholds
    }

    positive_threshold_counts = {
        f"positive_patches_zero_ge_{str(th).replace('.', 'p')}pct": 0
        for th in patch_zero_thresholds
    }

    patches_any_zero = 0
    positive_patches_any_zero = 0

    with rasterio.open(rtc_path) as rtc, rasterio.open(label_path) as label:
        for row in patch_rows:
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

            rtc_window = rtc.read([1, 2], window=window, masked=True)
            label_window = label.read(1, window=window, masked=True)

            finite_both, zero_both = compute_rtc_zero_mask(rtc_window, zero_epsilon)
            label_positive = compute_label_positive_mask(label_window)

            total_pixels = int(height * width)
            finite_pixels = int(np.count_nonzero(finite_both))
            zero_pixels = int(np.count_nonzero(zero_both))
            label_positive_pixels = int(np.count_nonzero(label_positive))
            zero_label_overlap_pixels = int(np.count_nonzero(zero_both & label_positive))

            finite_percent = 100.0 * finite_pixels / total_pixels if total_pixels else 0.0
            zero_percent = 100.0 * zero_pixels / total_pixels if total_pixels else 0.0

            has_positive = label_positive_pixels > 0
            has_any_zero = zero_pixels > 0

            if has_any_zero:
                patches_any_zero += 1

            if has_positive and has_any_zero:
                positive_patches_any_zero += 1

            flags: Dict[str, object] = {}

            for threshold in patch_zero_thresholds:
                key = f"zero_ge_{str(threshold).replace('.', 'p')}pct"
                value = zero_percent >= threshold
                flags[key] = bool_to_text(value)

                if value:
                    threshold_counts[f"patches_zero_ge_{str(threshold).replace('.', 'p')}pct"] += 1

                    if has_positive:
                        positive_threshold_counts[f"positive_patches_zero_ge_{str(threshold).replace('.', 'p')}pct"] += 1

            patch_row: Dict[str, object] = {
                "patch_id": row["patch_id"],
                "city": city,
                "region": row.get("region", ""),
                "row_start": row_start,
                "col_start": col_start,
                "height": height,
                "width": width,
                "patch_label_binary": safe_int(row.get("patch_label_binary", 0)),
                "label_positive_pixels": label_positive_pixels,
                "rtc_finite_pixels": finite_pixels,
                "rtc_finite_percent": round(finite_percent, 8),
                "rtc_all_zero_pixels": zero_pixels,
                "rtc_all_zero_percent": round(zero_percent, 8),
                "zero_label_overlap_pixels": zero_label_overlap_pixels,
                "has_any_rtc_zero": bool_to_text(has_any_zero),
                "has_positive_label": bool_to_text(has_positive),
            }

            patch_row.update(flags)
            patch_outputs.append(patch_row)

    summary = {
        "patches": len(patch_rows),
        "positive_patches": sum(1 for r in patch_outputs if r["patch_label_binary"] == 1),
        "patches_any_zero": patches_any_zero,
        "positive_patches_any_zero": positive_patches_any_zero,
    }

    summary.update(threshold_counts)
    summary.update(positive_threshold_counts)

    return patch_outputs, summary


# ---------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------

def validate_city(
    city: str,
    *,
    instance_root: Path,
    patch_rows: List[Dict[str, str]],
    finalization_by_city: Dict[str, Dict[str, str]],
    zero_epsilon: float,
    transform_tolerance: float,
    patch_zero_thresholds: Sequence[float],
    zero_warning_threshold_percent: float,
    compute_patch_stats: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    s2_path = find_s2_reference(instance_root, city)
    label_path = find_label_reference(instance_root, city)
    rtc_path = find_rtc_ready(instance_root, city)

    final_row = finalization_by_city.get(city, {})
    finalization_route = final_row.get("finalization_route", "")
    finalizer_invalid = final_row.get("max_invalid_percent_before_fill", "")

    grid = validate_city_grid(
        city=city,
        rtc_path=rtc_path,
        s2_path=s2_path,
        label_path=label_path,
        transform_tolerance=transform_tolerance,
    )

    city_stats = compute_city_rtc_stats(
        rtc_path=rtc_path,
        label_path=label_path,
        zero_epsilon=zero_epsilon,
    )

    patch_outputs: List[Dict[str, object]] = []
    patch_summary: Dict[str, object] = {
        "patches": len(patch_rows),
        "positive_patches": "",
        "patches_any_zero": "",
        "positive_patches_any_zero": "",
    }

    if compute_patch_stats:
        patch_outputs, patch_summary = compute_patch_stats_for_city(
            city=city,
            patch_rows=patch_rows,
            rtc_path=rtc_path,
            label_path=label_path,
            zero_epsilon=zero_epsilon,
            patch_zero_thresholds=patch_zero_thresholds,
        )

    notes: List[str] = []

    status = "ok"

    if grid["open_error"]:
        status = "failed"
        notes.append(f"Open/grid error: {grid['open_error']}")

    if not bool(grid["band_count_ok"]):
        status = "failed"
        notes.append("RTC band count is not 2.")

    if not bool(grid["grid_matches_s2"]):
        status = "failed"
        notes.append("RTC grid does not match S2.")

    if not bool(grid["label_grid_matches_s2"]):
        status = "failed"
        notes.append("Label grid does not match S2.")

    if safe_float(city_stats["rtc_finite_percent"]) < 100.0:
        notes.append("RTC has non-finite or masked pixels.")

    if safe_float(city_stats["rtc_all_zero_percent"]) > zero_warning_threshold_percent:
        notes.append(
            "RTC all-zero VV/VH percentage exceeds warning threshold; likely coverage/fill issue."
        )

    row: Dict[str, object] = {
        "city": city,
        "status": status,
        "finalization_route": finalization_route,
        "finalizer_max_invalid_percent_before_fill": finalizer_invalid,
        "rtc_path": path_to_str(rtc_path),
        "s2_reference_path": path_to_str(s2_path),
        "label_path": path_to_str(label_path),
    }

    row.update(grid)
    row.update(city_stats)
    row.update(patch_summary)

    row["notes"] = " | ".join(notes)

    return row, patch_outputs


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    *,
    instance_root: Path,
    rtc_root: Path,
    patch_metadata_csv: Path,
    finalization_csv: Optional[Path],
    city_rows: List[Dict[str, object]],
    patch_rows: List[Dict[str, object]],
    args: argparse.Namespace,
    city_csv: Path,
    patch_csv: Path,
    json_path: Path,
    md_path: Path,
) -> Dict[str, object]:
    n = len(city_rows)

    n_grid_ok = sum(1 for r in city_rows if bool(r.get("grid_matches_s2", False)))
    n_bands_ok = sum(1 for r in city_rows if bool(r.get("band_count_ok", False)))
    n_finite_lt_100 = sum(1 for r in city_rows if safe_float(r.get("rtc_finite_percent", 0)) < 100.0)
    n_zero_gt = sum(
        1 for r in city_rows
        if safe_float(r.get("rtc_all_zero_percent", 0)) > args.zero_warning_threshold_percent
    )
    n_failed = sum(1 for r in city_rows if r.get("status") != "ok")

    total_pixels = sum(safe_int(r.get("rtc_total_pixels", 0)) for r in city_rows)
    total_zero = sum(safe_int(r.get("rtc_all_zero_pixels", 0)) for r in city_rows)
    total_label_positive = sum(safe_int(r.get("label_positive_pixels", 0)) for r in city_rows)
    total_label_zero_overlap = sum(
        safe_int(r.get("label_positive_pixels_affected_by_zero", 0))
        for r in city_rows
    )

    total_zero_percent = 100.0 * total_zero / total_pixels if total_pixels else 0.0
    total_label_zero_overlap_percent = (
        100.0 * total_label_zero_overlap / total_label_positive
        if total_label_positive else 0.0
    )

    route_counts = Counter(str(r.get("finalization_route", "")) for r in city_rows)

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "rtc_root": path_to_str(rtc_root),
        "patch_metadata_csv": path_to_str(patch_metadata_csv),
        "finalization_csv": path_to_str(finalization_csv),
        "n_cities_validated": n,
        "n_cities_grid_ok": n_grid_ok,
        "n_cities_band_count_ok": n_bands_ok,
        "n_cities_finite_lt_100": n_finite_lt_100,
        "n_cities_zero_gt_threshold": n_zero_gt,
        "n_cities_failed": n_failed,
        "total_rtc_pixels": total_pixels,
        "total_rtc_all_zero_pixels": total_zero,
        "total_rtc_all_zero_percent": round(total_zero_percent, 8),
        "total_label_positive_pixels": total_label_positive,
        "total_label_positive_pixels_affected_by_zero": total_label_zero_overlap,
        "total_label_positive_pixels_affected_by_zero_percent": round(total_label_zero_overlap_percent, 8),
        "n_patch_rows": len(patch_rows),
        "route_counts": dict(sorted(route_counts.items())),
        "parameters": {
            "zero_epsilon": args.zero_epsilon,
            "zero_warning_threshold_percent": args.zero_warning_threshold_percent,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "transform_tolerance": args.transform_tolerance,
            "compute_patch_stats": not args.no_patch_stats,
            "patch_zero_thresholds": args.patch_zero_thresholds,
        },
        "outputs": {
            "city_csv": path_to_str(city_csv),
            "patch_csv": path_to_str(patch_csv),
            "json": path_to_str(json_path),
            "markdown": path_to_str(md_path),
        },
        "city_rows": city_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate finalized S1 RTC ready rasters against S2, labels, and patch metadata."
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
        "--patch-metadata-csv",
        type=Path,
        default=None,
        help=(
            "Optional patch metadata CSV. "
            "Default: <instance-root>/metadata/instance_C_patches/"
            "patch_metadata_ps<patch-size>_st<stride>_<edge-mode>.csv"
        ),
    )

    parser.add_argument(
        "--finalization-csv",
        type=Path,
        default=None,
        help=(
            "Optional finalization CSV. "
            "Default: <instance-root>/metadata/rtc_processing/s1_rtc_ready_finalization.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. "
            "Default: <instance-root>/metadata/rtc_processing"
        ),
    )

    parser.add_argument(
        "--zero-epsilon",
        type=float,
        default=1e-6,
        help="Tolerance for treating VV and VH as zero. Default: 1e-6.",
    )

    parser.add_argument(
        "--zero-warning-threshold-percent",
        type=float,
        default=1.0,
        help="Warn when city-level all-zero VV/VH percentage exceeds this threshold. Default: 1 percent.",
    )

    parser.add_argument(
        "--patch-zero-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 5.0, 25.0, 50.0],
        help="Patch-level zero-percent thresholds to summarize. Default: 0 1 5 25 50.",
    )

    parser.add_argument(
        "--transform-tolerance",
        type=float,
        default=0.0,
        help="Affine transform tolerance. Default: 0.0 means exact match.",
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Optional subset of cities to validate.",
    )

    parser.add_argument(
        "--no-patch-stats",
        action="store_true",
        help="Skip patch-level validation CSV. City-level validation still runs.",
    )

    parser.add_argument(
        "--fail-if-city-failed",
        action="store_true",
        help="Exit with code 2 if any city fails hard validation.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing validation outputs.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root
    rtc_root = instance_root / "s1_rtc_ready"

    output_dir: Path = args.output_dir or (
        instance_root / "metadata" / "rtc_processing"
    )

    patch_metadata_csv: Path = args.patch_metadata_csv or default_patch_metadata_path(
        instance_root=instance_root,
        patch_size=args.patch_size,
        stride=args.stride,
        edge_mode=args.edge_mode,
    )

    finalization_csv: Path = args.finalization_csv or (
        output_dir / "s1_rtc_ready_finalization.csv"
    )

    city_csv = output_dir / "s1_rtc_ready_validation_city.csv"
    patch_csv = output_dir / f"s1_rtc_ready_validation_patches_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    json_path = output_dir / "s1_rtc_ready_validation.json"
    md_path = output_dir / "s1_rtc_ready_validation.md"

    log("STEP", "Validating S1 RTC ready rasters.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"RTC root:      {path_to_str(rtc_root)}")
    log("INFO", f"Patch CSV:     {path_to_str(patch_metadata_csv)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not rtc_root.exists():
        fail(f"RTC root does not exist: {path_to_str(rtc_root)}")

    patch_rows_input = read_csv_rows(patch_metadata_csv)
    patch_rows_by_city = group_patch_rows_by_city(patch_rows_input)

    finalization_by_city = read_finalization_by_city(finalization_csv)

    cities = sorted(patch_rows_by_city.keys())

    if args.cities:
        requested = set(normalize_city(city) for city in args.cities)
        missing = sorted(requested - set(cities))

        if missing:
            fail(
                "Requested cities were not found in patch metadata:\n"
                + "\n".join(f"  - {city}" for city in missing)
            )

        cities = [city for city in cities if city in requested]
        log("WARN", f"City subset enabled: {', '.join(cities)}")

    city_outputs: List[Dict[str, object]] = []
    patch_outputs_all: List[Dict[str, object]] = []

    for city in cities:
        log("STEP", f"Validating city: {city}")

        city_row, patch_rows = validate_city(
            city=city,
            instance_root=instance_root,
            patch_rows=patch_rows_by_city[city],
            finalization_by_city=finalization_by_city,
            zero_epsilon=args.zero_epsilon,
            transform_tolerance=args.transform_tolerance,
            patch_zero_thresholds=args.patch_zero_thresholds,
            zero_warning_threshold_percent=args.zero_warning_threshold_percent,
            compute_patch_stats=not args.no_patch_stats,
        )

        city_outputs.append(city_row)
        patch_outputs_all.extend(patch_rows)

        log(
            "OK" if city_row["status"] == "ok" else "WARN",
            f"{city}: status={city_row['status']}, "
            f"grid_ok={city_row['grid_matches_s2']}, "
            f"bands_ok={city_row['band_count_ok']}, "
            f"finite={city_row['rtc_finite_percent']}%, "
            f"all_zero={city_row['rtc_all_zero_percent']}%, "
            f"label_zero_overlap={city_row['label_positive_pixels_affected_by_zero_percent']}%",
        )

    summary = build_summary(
        instance_root=instance_root,
        rtc_root=rtc_root,
        patch_metadata_csv=patch_metadata_csv,
        finalization_csv=finalization_csv,
        city_rows=city_outputs,
        patch_rows=patch_outputs_all,
        args=args,
        city_csv=city_csv,
        patch_csv=patch_csv,
        json_path=json_path,
        md_path=md_path,
    )

    log("STEP", "Writing validation outputs.")

    write_csv(city_csv, city_outputs, overwrite=args.overwrite)

    if not args.no_patch_stats:
        write_csv(patch_csv, patch_outputs_all, overwrite=args.overwrite)

    write_json(json_path, summary, overwrite=args.overwrite)
    write_markdown(md_path, summary, city_outputs, overwrite=args.overwrite)

    log("OK", f"Wrote city CSV:  {path_to_str(city_csv)}")

    if not args.no_patch_stats:
        log("OK", f"Wrote patch CSV: {path_to_str(patch_csv)}")

    log("OK", f"Wrote JSON:      {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:  {path_to_str(md_path)}")

    log("STEP", "Final summary.")
    log("OK", f"Cities validated: {summary['n_cities_validated']}")
    log("OK", f"Cities grid OK: {summary['n_cities_grid_ok']}")
    log("OK", f"Cities band count OK: {summary['n_cities_band_count_ok']}")
    log("OK", f"Cities failed: {summary['n_cities_failed']}")
    log("OK", f"Cities zero > threshold: {summary['n_cities_zero_gt_threshold']}")
    log("OK", f"Total RTC all-zero percent: {summary['total_rtc_all_zero_percent']}")
    log("OK", f"Total label-positive pixels affected by zero percent: {summary['total_label_positive_pixels_affected_by_zero_percent']}")

    if summary["n_cities_failed"] > 0 and args.fail_if_city_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()