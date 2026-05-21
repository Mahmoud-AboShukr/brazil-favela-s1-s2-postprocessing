#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
11_compute_s1_snap_rtc_statistics_224.py

Option B: patch-based raw Sentinel-1 statistics for SNAP-GRD vs RTC.

Purpose
-------
Compute raw VV/VH pixel-value statistics for the exact 224x224 patch set used
in the CROMA comparison, then compare these statistics against the published
SSL4EO-S12 Sentinel-1 reference statistics.

This answers Thomas's suggested sanity-check question:

    Are our SNAP-GRD and RTC distributions in a physically plausible range
    compared with a known large-scale Sentinel-1 reference dataset?

Important
---------
This is NOT an embedding analysis and NOT a model-performance metric.

It directly inspects the SAR raster values used by the CROMA patch pipeline.

Comparison reference
--------------------
SSL4EO-S12 Sentinel-1 GRD statistics, dB scale:

    VV mean = -12.59
    VV std  =   5.26

    VH mean = -20.26
    VH std  =   5.91

Method
------
The script reads the CROMA comparison manifest:

    metadata/croma_probing/croma_comparison_manifest_ps224_st112_cover.csv

and selects only:

    s1_snap_vv_vh
    s1_rtc_vv_vh

For each patch row, it reads the SAR raster window and bands 1 and 2:

    band 1 = VV
    band 2 = VH

It first performs a small scale-detection pass to check whether the rasters
look like dB values or linear sigma0 values.

Then it computes exact streaming mean/std/min/max over valid patch pixels.
Percentiles are estimated from a large random sample because storing every
pixel from every overlapping patch would be unnecessarily large.

Outputs
-------
<instance-root>/metadata/croma_probing/s1_statistics_patch_based/

    s1_patch_based_scale_detection_ps224_st112_cover.csv
    s1_patch_based_global_statistics_ps224_st112_cover.csv
    s1_patch_based_city_statistics_ps224_st112_cover.csv
    s1_patch_based_ssl4eo_comparison_ps224_st112_cover.csv
    s1_patch_based_statistics_summary_ps224_st112_cover.json
    s1_patch_based_statistics_summary_ps224_st112_cover.md

Optional figures:

    figures/s1_patch_based_histogram_vv_ps224_st112_cover.png
    figures/s1_patch_based_histogram_vh_ps224_st112_cover.png
    figures/s1_patch_based_city_mean_vv_ps224_st112_cover.png
    figures/s1_patch_based_city_mean_vh_ps224_st112_cover.png

Example
-------
python src/croma_probing/11_compute_s1_snap_rtc_statistics_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --make-figures `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------
# Reference statistics
# ---------------------------------------------------------------------

SSL4EO_S1_REFERENCE = {
    "VV": {
        "mean": -12.59,
        "std": 5.26,
    },
    "VH": {
        "mean": -20.26,
        "std": 5.91,
    },
}


PRODUCT_MODALITIES = {
    "SNAP-GRD": "s1_snap_vv_vh",
    "RTC": "s1_rtc_vv_vh",
}


BAND_INDEX_TO_NAME = {
    1: "VV",
    2: "VH",
}


# ---------------------------------------------------------------------
# Logging and utilities
# ---------------------------------------------------------------------

def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    log("ERROR", message)
    raise SystemExit(exit_code)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if text == "":
            return default
        out = float(text)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def round_float(value: float, digits: int = 8) -> float:
    return round(safe_float(value, 0.0), digits)


def split_semicolon_ints(value: object, default: Sequence[int]) -> List[int]:
    text = str(value).strip()

    if text == "":
        return list(default)

    parts = text.replace(",", ";").split(";")
    out: List[int] = []

    for part in parts:
        part = part.strip()
        if part == "":
            continue
        try:
            out.append(int(float(part)))
        except Exception:
            pass

    if not out:
        return list(default)

    return out


# ---------------------------------------------------------------------
# CSV / JSON / Markdown
# ---------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        fail(f"Input CSV does not exist: {path_to_str(path)}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        fail(f"Input CSV is empty: {path_to_str(path)}")

    return rows


def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    overwrite: bool,
    fieldnames: Optional[List[str]] = None,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    if fieldnames is None:
        if not rows:
            fail(f"No rows to write and no fieldnames were provided: {path_to_str(path)}")
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
    *,
    summary: Dict[str, object],
    scale_rows: List[Dict[str, object]],
    global_rows: List[Dict[str, object]],
    ssl4eo_rows: List[Dict[str, object]],
    output_paths: Dict[str, Optional[Path]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Patch-Based SNAP-GRD vs RTC Sentinel-1 Statistics")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Manifest: `{summary['manifest_path']}`")
    lines.append(f"- Output directory: `{summary['output_dir']}`")
    lines.append(f"- Patch size: `{summary['patch_size']}`")
    lines.append(f"- Stride: `{summary['stride']}`")
    lines.append(f"- Edge mode: `{summary['edge_mode']}`")
    lines.append(f"- Total manifest rows used: `{summary['total_manifest_rows_used']}`")
    lines.append("")

    lines.append("### Main conclusion")
    lines.append("")
    lines.append(summary["main_conclusion"])
    lines.append("")

    lines.append("## Methodological note")
    lines.append("")
    lines.append("This is a patch-based distribution check. The statistics are computed over the exact 224 by 224 patch windows used in the CROMA comparison. Because the patches use stride 112, overlapping pixels are counted more than once. This is intentional for Option B, because the objective is to describe the distribution seen by the patch-based ML/CROMA pipeline rather than the unique full-city raster distribution.")
    lines.append("")
    lines.append("The SSL4EO-S12 comparison is a broad plausibility check, not a strict matching criterion. SSL4EO-S12 is global and multi-seasonal, while this dataset is focused on Brazilian urban/favela areas.")
    lines.append("")

    lines.append("## Scale detection")
    lines.append("")
    lines.append("| product | band | inferred scale | convert to dB | raw mean | raw median | raw p01 | raw p99 | notes |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|")

    for row in scale_rows:
        lines.append(
            f"| {row['product']} | "
            f"{row['band']} | "
            f"{row['inferred_scale']} | "
            f"{row['convert_to_db']} | "
            f"{row['raw_mean']} | "
            f"{row['raw_median']} | "
            f"{row['raw_p01']} | "
            f"{row['raw_p99']} | "
            f"{row['notes']} |"
        )

    lines.append("")
    lines.append("## Global patch-based statistics")
    lines.append("")
    lines.append("| product | band | scale used | valid pixels | invalid % | mean | median | std | p01 | p05 | p95 | p99 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in global_rows:
        lines.append(
            f"| {row['product']} | "
            f"{row['band']} | "
            f"{row['scale_used']} | "
            f"{row['valid_pixel_count']} | "
            f"{row['invalid_percent']} | "
            f"{row['mean']} | "
            f"{row['median']} | "
            f"{row['std']} | "
            f"{row['p01']} | "
            f"{row['p05']} | "
            f"{row['p95']} | "
            f"{row['p99']} |"
        )

    lines.append("")
    lines.append("## Comparison with SSL4EO-S12 Sentinel-1 statistics")
    lines.append("")
    lines.append("| product | band | our mean | SSL4EO mean | mean diff | our std | SSL4EO std | std ratio | plausibility |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")

    for row in ssl4eo_rows:
        lines.append(
            f"| {row['product']} | "
            f"{row['band']} | "
            f"{row['our_mean']} | "
            f"{row['ssl4eo_mean']} | "
            f"{row['mean_difference']} | "
            f"{row['our_std']} | "
            f"{row['ssl4eo_std']} | "
            f"{row['std_ratio_our_over_ssl4eo']} | "
            f"{row['plausibility_flag']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The key interpretation is whether the VV and VH values fall into a plausible Sentinel-1 dB range. For reference, SSL4EO-S12 reports VV mean -12.59 dB and std 5.26, and VH mean -20.26 dB and std 5.91. Exact agreement is not expected because SSL4EO-S12 is global and multi-seasonal, while this dataset is Brazil-urban/favela-focused.")
    lines.append("")
    lines.append("If one product has a very different mean, very compressed standard deviation, extreme tails, or a detected linear rather than dB scale, this would indicate a preprocessing or scaling issue that should be investigated before further modelling.")
    lines.append("")

    if output_paths.get("histogram_vv") is not None:
        lines.append("## Optional generated figures")
        lines.append("")
        for key in ["histogram_vv", "histogram_vh", "city_mean_vv", "city_mean_vh"]:
            if output_paths.get(key) is not None:
                lines.append(f"- `{key}`: `{path_to_str(output_paths[key])}`")
        lines.append("")

    lines.append("## Output files")
    lines.append("")
    for key, value in output_paths.items():
        if value is not None:
            lines.append(f"- `{key}`: `{path_to_str(value)}`")

    lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------

def find_column(fieldnames: Sequence[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lower_to_original = {name.lower(): name for name in fieldnames}

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    if required:
        fail(
            "Could not find required column. Tried candidates:\n"
            f"  {candidates}\n"
            f"Available columns:\n"
            f"  {list(fieldnames)}"
        )

    return None


def default_manifest_path(instance_root: Path, stem: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / f"croma_comparison_manifest_{stem}.csv"
    )


def load_product_manifest_rows(
    manifest_path: Path,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, str]]:
    rows = read_csv_rows(manifest_path)
    fieldnames = list(rows[0].keys())

    columns = {
        "modality": find_column(fieldnames, ["modality"]),
        "sar_path": find_column(
            fieldnames,
            [
                "sar_path",
                "s1_path",
                "s1_raster_path",
                "raster_path",
                "image_path",
                "input_path",
                "path",
            ],
        ),
        "row_off": find_column(
            fieldnames,
            [
                "row_off",
                "row_start",
                "window_row_off",
                "patch_row_off",
                "row",
                "y_off",
                "y",
            ],
        ),
        "col_off": find_column(
            fieldnames,
            [
                "col_off",
                "col_start",
                "window_col_off",
                "patch_col_off",
                "col",
                "x_off",
                "x",
            ],
        ),
        "height": find_column(
            fieldnames,
            [
                "height",
                "patch_height",
                "window_height",
                "h",
            ],
            required=False,
        ),
        "width": find_column(
            fieldnames,
            [
                "width",
                "patch_width",
                "window_width",
                "w",
            ],
            required=False,
        ),
        "city": find_column(fieldnames, ["city"]),
        "region": find_column(fieldnames, ["region"], required=False),
        "patch_id": find_column(fieldnames, ["patch_id", "patch_uid", "id"], required=False),
        "sar_band_indices": find_column(
            fieldnames,
            [
                "sar_band_indices",
                "s1_band_indices",
                "band_indices",
                "bands",
            ],
            required=False,
        ),
    }

    product_rows: Dict[str, List[Dict[str, str]]] = {
        product: []
        for product in PRODUCT_MODALITIES
    }

    modality_col = columns["modality"]

    for row in rows:
        modality = str(row[modality_col]).strip()

        for product, expected_modality in PRODUCT_MODALITIES.items():
            if modality == expected_modality:
                product_rows[product].append(row)

    for product, selected_rows in product_rows.items():
        if not selected_rows:
            fail(
                f"No manifest rows found for product={product}, "
                f"expected modality={PRODUCT_MODALITIES[product]}"
            )

        log("OK", f"{product}: manifest rows={len(selected_rows)}")

    return product_rows, columns


def row_to_window(row: Dict[str, str], columns: Dict[str, Optional[str]], patch_size: int) -> Window:
    row_off = safe_int(row[columns["row_off"]])
    col_off = safe_int(row[columns["col_off"]])

    if columns["height"] is not None:
        height = safe_int(row[columns["height"]], patch_size)
    else:
        height = patch_size

    if columns["width"] is not None:
        width = safe_int(row[columns["width"]], patch_size)
    else:
        width = patch_size

    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


def row_to_band_indices(row: Dict[str, str], columns: Dict[str, Optional[str]]) -> List[int]:
    if columns["sar_band_indices"] is None:
        return [1, 2]

    indices = split_semicolon_ints(row[columns["sar_band_indices"]], default=[1, 2])

    out = []
    for idx in indices:
        if idx in {1, 2}:
            out.append(idx)

    if not out:
        out = [1, 2]

    return out[:2]


# ---------------------------------------------------------------------
# Statistics containers
# ---------------------------------------------------------------------

class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_value = float("inf")
        self.max_value = float("-inf")
        self.invalid_count = 0
        self.total_count = 0

    def update_invalid_total(self, total_count: int, invalid_count: int) -> None:
        self.total_count += int(total_count)
        self.invalid_count += int(invalid_count)

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return

        batch_n = int(arr.size)
        batch_mean = float(np.mean(arr))
        batch_m2 = float(np.sum((arr - batch_mean) ** 2))

        self.min_value = min(self.min_value, float(np.min(arr)))
        self.max_value = max(self.max_value, float(np.max(arr)))

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total_n = self.n + batch_n
        delta = batch_mean - self.mean

        self.mean = self.mean + delta * batch_n / total_n
        self.m2 = self.m2 + batch_m2 + delta * delta * self.n * batch_n / total_n
        self.n = total_n

    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))

    def invalid_percent(self) -> float:
        if self.total_count <= 0:
            return 0.0
        return 100.0 * self.invalid_count / self.total_count

    def to_basic_dict(self) -> Dict[str, object]:
        return {
            "valid_pixel_count": int(self.n),
            "total_pixel_count": int(self.total_count),
            "invalid_pixel_count": int(self.invalid_count),
            "invalid_percent": round_float(self.invalid_percent(), 8),
            "mean": round_float(self.mean, 8),
            "std": round_float(self.std(), 8),
            "min": round_float(self.min_value if self.n > 0 else 0.0, 8),
            "max": round_float(self.max_value if self.n > 0 else 0.0, 8),
        }


class SampleStore:
    def __init__(self, max_samples: int, random_state: int) -> None:
        self.max_samples = int(max_samples)
        self.rng = np.random.default_rng(random_state)
        self.chunks: List[np.ndarray] = []

    def add(self, values: np.ndarray, per_window: int) -> None:
        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            return

        if per_window > 0 and arr.size > per_window:
            idx = self.rng.choice(arr.size, size=per_window, replace=False)
            arr = arr[idx]

        self.chunks.append(arr)

    def values(self) -> np.ndarray:
        if not self.chunks:
            return np.asarray([], dtype=np.float32)

        arr = np.concatenate(self.chunks)

        if self.max_samples > 0 and arr.size > self.max_samples:
            idx = self.rng.choice(arr.size, size=self.max_samples, replace=False)
            arr = arr[idx]

        return arr.astype(np.float32)


# ---------------------------------------------------------------------
# Raster reading and scale detection
# ---------------------------------------------------------------------

def read_patch_bands(
    row: Dict[str, str],
    columns: Dict[str, Optional[str]],
    patch_size: int,
) -> Dict[int, np.ndarray]:
    sar_path = Path(row[columns["sar_path"]])
    band_indices = row_to_band_indices(row, columns)
    window = row_to_window(row, columns, patch_size=patch_size)

    if not sar_path.exists():
        fail(f"SAR raster does not exist: {path_to_str(sar_path)}")

    out: Dict[int, np.ndarray] = {}

    with rasterio.open(sar_path) as src:
        for band_index in band_indices:
            if band_index > src.count:
                fail(
                    f"Requested band {band_index}, but raster has only {src.count} bands:\n"
                    f"  {path_to_str(sar_path)}"
                )

            data = src.read(
                band_index,
                window=window,
                masked=True,
                boundless=False,
            )

            if np.ma.isMaskedArray(data):
                arr = data.filled(np.nan).astype(np.float32)
            else:
                arr = np.asarray(data, dtype=np.float32)

            out[band_index] = arr

    return out


def valid_raw_values(
    arr: np.ndarray,
    *,
    treat_zero_as_invalid: bool,
) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32).reshape(-1)
    mask = np.isfinite(values)

    if treat_zero_as_invalid:
        mask &= values != 0

    return values[mask]


def summarize_array(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "std": 0.0,
        }

    return {
        "count": int(arr.size),
        "min": round_float(float(np.min(arr)), 8),
        "p01": round_float(float(np.quantile(arr, 0.01)), 8),
        "p05": round_float(float(np.quantile(arr, 0.05)), 8),
        "median": round_float(float(np.median(arr)), 8),
        "mean": round_float(float(np.mean(arr)), 8),
        "p95": round_float(float(np.quantile(arr, 0.95)), 8),
        "p99": round_float(float(np.quantile(arr, 0.99)), 8),
        "max": round_float(float(np.max(arr)), 8),
        "std": round_float(float(np.std(arr)), 8),
    }


def infer_scale(values: np.ndarray) -> Tuple[str, bool, str]:
    stats = summarize_array(values)

    p01 = safe_float(stats["p01"])
    p99 = safe_float(stats["p99"])
    median = safe_float(stats["median"])
    mean = safe_float(stats["mean"])
    min_value = safe_float(stats["min"])
    max_value = safe_float(stats["max"])

    if stats["count"] == 0:
        return "unknown_no_valid_values", False, "No valid values found during scale probe."

    if median < 0 and p01 > -100 and p99 < 50:
        return "db", False, "Values look like Sentinel-1 dB backscatter."

    if min_value >= 0 and p99 <= 5.0 and median < 1.5 and mean < 1.5:
        return "linear_sigma0_power", True, "Values look like linear sigma0 power; converting to dB using 10*log10(x)."

    if min_value >= 0 and max_value > 5.0:
        return "unknown_positive_large", False, "Values are positive but not clearly linear sigma0; leaving unchanged."

    return "unknown_assumed_db", False, "Scale is ambiguous; leaving unchanged and reporting this."


def detect_scales(
    product_rows: Dict[str, List[Dict[str, str]]],
    columns: Dict[str, Optional[str]],
    *,
    patch_size: int,
    scale_probe_max_patches: int,
    scale_probe_pixels_per_patch: int,
    treat_zero_as_invalid: bool,
    random_state: int,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str], Dict[str, object]]]:
    rng = np.random.default_rng(random_state)

    scale_rows: List[Dict[str, object]] = []
    scale_info: Dict[Tuple[str, str], Dict[str, object]] = {}

    for product, rows in product_rows.items():
        log("STEP", f"Scale detection for product: {product}")

        if scale_probe_max_patches > 0 and len(rows) > scale_probe_max_patches:
            idx = rng.choice(len(rows), size=scale_probe_max_patches, replace=False)
            probe_rows = [rows[int(i)] for i in idx]
        else:
            probe_rows = rows

        band_samples: Dict[str, List[np.ndarray]] = {
            "VV": [],
            "VH": [],
        }

        for row in probe_rows:
            band_arrays = read_patch_bands(row, columns, patch_size=patch_size)

            for band_index, arr in band_arrays.items():
                band_name = BAND_INDEX_TO_NAME.get(band_index)

                if band_name is None:
                    continue

                values = valid_raw_values(arr, treat_zero_as_invalid=treat_zero_as_invalid)

                if values.size == 0:
                    continue

                if scale_probe_pixels_per_patch > 0 and values.size > scale_probe_pixels_per_patch:
                    sample_idx = rng.choice(values.size, size=scale_probe_pixels_per_patch, replace=False)
                    values = values[sample_idx]

                band_samples[band_name].append(values.astype(np.float32))

        for band_name in ["VV", "VH"]:
            if band_samples[band_name]:
                values = np.concatenate(band_samples[band_name])
            else:
                values = np.asarray([], dtype=np.float32)

            inferred_scale, convert_to_db, notes = infer_scale(values)
            stats = summarize_array(values)

            row = {
                "product": product,
                "band": band_name,
                "raw_sample_count": stats["count"],
                "raw_min": stats["min"],
                "raw_p01": stats["p01"],
                "raw_p05": stats["p05"],
                "raw_median": stats["median"],
                "raw_mean": stats["mean"],
                "raw_p95": stats["p95"],
                "raw_p99": stats["p99"],
                "raw_max": stats["max"],
                "raw_std": stats["std"],
                "inferred_scale": inferred_scale,
                "convert_to_db": bool(convert_to_db),
                "notes": notes,
            }

            scale_rows.append(row)

            scale_info[(product, band_name)] = {
                "inferred_scale": inferred_scale,
                "convert_to_db": bool(convert_to_db),
                "notes": notes,
            }

            log(
                "OK",
                f"{product} {band_name}: scale={inferred_scale}, "
                f"convert_to_db={convert_to_db}, raw_mean={stats['mean']}, raw_p99={stats['p99']}",
            )

    return scale_rows, scale_info


def transform_values_to_comparison_scale(
    values: np.ndarray,
    *,
    convert_to_db: bool,
) -> Tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    mask = np.isfinite(arr)

    if convert_to_db:
        mask &= arr > 0

    invalid_count = int(arr.size - np.count_nonzero(mask))

    arr = arr[mask]

    if convert_to_db:
        arr = 10.0 * np.log10(arr)

    return arr.astype(np.float32), invalid_count


# ---------------------------------------------------------------------
# Main statistics computation
# ---------------------------------------------------------------------

def compute_patch_based_statistics(
    product_rows: Dict[str, List[Dict[str, str]]],
    columns: Dict[str, Optional[str]],
    scale_info: Dict[Tuple[str, str], Dict[str, object]],
    *,
    patch_size: int,
    percentile_sample_per_window: int,
    percentile_max_samples: int,
    treat_zero_as_invalid: bool,
    random_state: int,
    max_patches_per_product: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[Tuple[str, str], np.ndarray]]:
    rng = np.random.default_rng(random_state)

    global_stats: Dict[Tuple[str, str], RunningStats] = {}
    city_stats: Dict[Tuple[str, str, str], RunningStats] = {}
    sample_stores: Dict[Tuple[str, str], SampleStore] = {}

    for product in PRODUCT_MODALITIES:
        for band in ["VV", "VH"]:
            global_stats[(product, band)] = RunningStats()
            sample_stores[(product, band)] = SampleStore(
                max_samples=percentile_max_samples,
                random_state=random_state + hash((product, band)) % 100000,
            )

    for product, rows in product_rows.items():
        log("STEP", f"Computing patch-based statistics for product: {product}")

        if max_patches_per_product > 0:
            rows_to_use = rows[:max_patches_per_product]
            log("WARN", f"{product}: debug mode, using only {len(rows_to_use)} patches.")
        else:
            rows_to_use = rows

        for idx, row in enumerate(rows_to_use, start=1):
            if idx == 1 or idx % 500 == 0 or idx == len(rows_to_use):
                log("INFO", f"{product}: processing patch {idx}/{len(rows_to_use)}")

            city = str(row[columns["city"]]).strip()

            band_arrays = read_patch_bands(row, columns, patch_size=patch_size)

            for band_index, arr in band_arrays.items():
                band_name = BAND_INDEX_TO_NAME.get(band_index)

                if band_name is None:
                    continue

                raw_values = np.asarray(arr, dtype=np.float32).reshape(-1)
                raw_mask = np.isfinite(raw_values)

                if treat_zero_as_invalid:
                    raw_mask &= raw_values != 0

                raw_values = raw_values[raw_mask]

                convert_to_db = bool(scale_info[(product, band_name)]["convert_to_db"])

                values, conversion_invalid = transform_values_to_comparison_scale(
                    raw_values,
                    convert_to_db=convert_to_db,
                )

                total_count = int(arr.size)
                invalid_count = int(total_count - values.size)

                global_stats[(product, band_name)].update_invalid_total(
                    total_count=total_count,
                    invalid_count=invalid_count,
                )
                global_stats[(product, band_name)].update(values)

                city_key = (product, band_name, city)
                if city_key not in city_stats:
                    city_stats[city_key] = RunningStats()

                city_stats[city_key].update_invalid_total(
                    total_count=total_count,
                    invalid_count=invalid_count,
                )
                city_stats[city_key].update(values)

                sample_stores[(product, band_name)].add(
                    values,
                    per_window=percentile_sample_per_window,
                )

    sample_values: Dict[Tuple[str, str], np.ndarray] = {
        key: store.values()
        for key, store in sample_stores.items()
    }

    global_rows: List[Dict[str, object]] = []

    for product in PRODUCT_MODALITIES:
        for band in ["VV", "VH"]:
            stats = global_stats[(product, band)].to_basic_dict()
            samples = sample_values[(product, band)]
            pct = summarize_array(samples)

            scale = scale_info[(product, band)]

            global_rows.append(
                {
                    "product": product,
                    "band": band,
                    "scale_used": "dB" if not scale["convert_to_db"] else "converted_linear_to_dB",
                    "inferred_input_scale": scale["inferred_scale"],
                    "converted_to_db": bool(scale["convert_to_db"]),
                    "valid_pixel_count": stats["valid_pixel_count"],
                    "total_pixel_count": stats["total_pixel_count"],
                    "invalid_pixel_count": stats["invalid_pixel_count"],
                    "invalid_percent": stats["invalid_percent"],
                    "mean": stats["mean"],
                    "median": pct["median"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "p01": pct["p01"],
                    "p05": pct["p05"],
                    "p25": round_float(float(np.quantile(samples, 0.25)), 8) if samples.size else 0.0,
                    "p75": round_float(float(np.quantile(samples, 0.75)), 8) if samples.size else 0.0,
                    "p95": pct["p95"],
                    "p99": pct["p99"],
                    "max": stats["max"],
                    "percentile_sample_count": pct["count"],
                }
            )

    city_rows: List[Dict[str, object]] = []

    for (product, band, city), stats_obj in sorted(city_stats.items()):
        stats = stats_obj.to_basic_dict()
        scale = scale_info[(product, band)]

        city_rows.append(
            {
                "product": product,
                "band": band,
                "city": city,
                "scale_used": "dB" if not scale["convert_to_db"] else "converted_linear_to_dB",
                "inferred_input_scale": scale["inferred_scale"],
                "converted_to_db": bool(scale["convert_to_db"]),
                "valid_pixel_count": stats["valid_pixel_count"],
                "total_pixel_count": stats["total_pixel_count"],
                "invalid_pixel_count": stats["invalid_pixel_count"],
                "invalid_percent": stats["invalid_percent"],
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
            }
        )

    return global_rows, city_rows, sample_values


def build_ssl4eo_comparison_rows(
    global_rows: List[Dict[str, object]],
    *,
    plausible_mean_tolerance_db: float,
    plausible_std_min: float,
    plausible_std_max: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for row in global_rows:
        product = str(row["product"])
        band = str(row["band"])

        ref = SSL4EO_S1_REFERENCE[band]

        our_mean = safe_float(row["mean"])
        our_std = safe_float(row["std"])

        mean_difference = our_mean - ref["mean"]
        std_difference = our_std - ref["std"]

        std_ratio = our_std / ref["std"] if ref["std"] != 0 else 0.0

        plausible_mean = abs(mean_difference) <= plausible_mean_tolerance_db
        plausible_std = plausible_std_min <= our_std <= plausible_std_max

        if plausible_mean and plausible_std:
            flag = "broadly_plausible"
        elif not plausible_mean and plausible_std:
            flag = "mean_shift_check_needed"
        elif plausible_mean and not plausible_std:
            flag = "std_range_check_needed"
        else:
            flag = "distribution_check_needed"

        rows.append(
            {
                "product": product,
                "band": band,
                "our_mean": round_float(our_mean, 8),
                "ssl4eo_mean": ref["mean"],
                "mean_difference": round_float(mean_difference, 8),
                "our_std": round_float(our_std, 8),
                "ssl4eo_std": ref["std"],
                "std_difference": round_float(std_difference, 8),
                "std_ratio_our_over_ssl4eo": round_float(std_ratio, 8),
                "plausible_mean_tolerance_db": plausible_mean_tolerance_db,
                "plausible_std_min": plausible_std_min,
                "plausible_std_max": plausible_std_max,
                "plausibility_flag": flag,
                "interpretation": (
                    "Broad sanity check against SSL4EO-S12 S1 GRD dB statistics; "
                    "not expected to match exactly because domains differ."
                ),
            }
        )

    return rows


def build_main_conclusion(scale_rows: List[Dict[str, object]], ssl4eo_rows: List[Dict[str, object]]) -> str:
    scale_problems = [
        row for row in scale_rows
        if str(row["inferred_scale"]).startswith("unknown")
    ]

    plausible = [
        row for row in ssl4eo_rows
        if row["plausibility_flag"] == "broadly_plausible"
    ]

    total = len(ssl4eo_rows)

    if not scale_problems and len(plausible) == total:
        return (
            "The patch-based SNAP-GRD and RTC VV/VH distributions appear broadly plausible "
            "relative to the SSL4EO-S12 Sentinel-1 dB reference statistics. The scale-detection "
            "step did not identify obvious linear-vs-dB inconsistencies."
        )

    if scale_problems and len(plausible) == total:
        return (
            "The final distributions appear broadly plausible relative to SSL4EO-S12, but at least "
            "one product/band had ambiguous scale detection and should be manually inspected."
        )

    if len(plausible) > 0:
        return (
            "Some product/band distributions are broadly plausible relative to SSL4EO-S12, while others "
            "show mean or standard-deviation shifts that should be inspected. This does not automatically "
            "invalidate the products because SSL4EO-S12 is global/multi-seasonal and our dataset is "
            "Brazil-urban/favela-focused."
        )

    return (
        "The patch-based distributions differ substantially from the broad SSL4EO-S12 reference range. "
        "This may reflect domain differences, but the product scale and preprocessing should be checked "
        "before using these statistics as supporting evidence."
    )


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def make_histogram_figure(
    *,
    sample_values: Dict[Tuple[str, str], np.ndarray],
    band: str,
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        log("WARN", "matplotlib is not installed; skipping histogram figure.")
        return None

    ensure_output_can_be_written(output_path, overwrite)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    for product in PRODUCT_MODALITIES:
        values = sample_values.get((product, band), np.asarray([], dtype=np.float32))
        values = values[np.isfinite(values)]

        if values.size == 0:
            continue

        ax.hist(values, bins=80, alpha=0.55, density=True, label=product)

    ref_mean = SSL4EO_S1_REFERENCE[band]["mean"]
    ax.axvline(ref_mean, linestyle="--", linewidth=1.5, label=f"SSL4EO {band} mean")

    ax.set_title(f"Patch-based Sentinel-1 {band} distribution")
    ax.set_xlabel(f"{band} backscatter value, dB scale")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def make_city_mean_figure(
    *,
    city_rows: List[Dict[str, object]],
    band: str,
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        log("WARN", "matplotlib is not installed; skipping city mean figure.")
        return None

    rows = [row for row in city_rows if row["band"] == band]

    if not rows:
        return None

    cities = sorted(set(str(row["city"]) for row in rows))

    snap_by_city = {
        str(row["city"]): safe_float(row["mean"])
        for row in rows
        if row["product"] == "SNAP-GRD"
    }

    rtc_by_city = {
        str(row["city"]): safe_float(row["mean"])
        for row in rows
        if row["product"] == "RTC"
    }

    ensure_output_can_be_written(output_path, overwrite)

    x = list(range(len(cities)))
    width = 0.38

    snap_values = [snap_by_city.get(city, float("nan")) for city in cities]
    rtc_values = [rtc_by_city.get(city, float("nan")) for city in cities]

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    ax.bar([i - width / 2 for i in x], snap_values, width, label="SNAP-GRD")
    ax.bar([i + width / 2 for i in x], rtc_values, width, label="RTC")

    ref_mean = SSL4EO_S1_REFERENCE[band]["mean"]
    ax.axhline(ref_mean, linestyle="--", linewidth=1.5, label=f"SSL4EO {band} mean")

    ax.set_title(f"City-level mean {band} backscatter")
    ax.set_ylabel(f"Mean {band}, dB scale")
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=60, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------
# Summary payload
# ---------------------------------------------------------------------

def build_summary_payload(
    *,
    instance_root: Path,
    manifest_path: Path,
    output_dir: Path,
    scale_rows: List[Dict[str, object]],
    global_rows: List[Dict[str, object]],
    city_rows: List[Dict[str, object]],
    ssl4eo_rows: List[Dict[str, object]],
    main_conclusion: str,
    output_paths: Dict[str, Optional[Path]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "created_utc": now_utc(),
        "status": "passed",
        "instance_root": path_to_str(instance_root),
        "manifest_path": path_to_str(manifest_path),
        "output_dir": path_to_str(output_dir),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "edge_mode": args.edge_mode,
        "total_manifest_rows_used": int(sum(safe_int(row["manifest_rows"]) for row in summarize_manifest_count_rows(scale_rows))),
        "main_conclusion": main_conclusion,
        "ssl4eo_reference": SSL4EO_S1_REFERENCE,
        "parameters": {
            "scale_probe_max_patches": args.scale_probe_max_patches,
            "scale_probe_pixels_per_patch": args.scale_probe_pixels_per_patch,
            "percentile_sample_per_window": args.percentile_sample_per_window,
            "percentile_max_samples": args.percentile_max_samples,
            "treat_zero_as_invalid": bool(args.treat_zero_as_invalid),
            "plausible_mean_tolerance_db": args.plausible_mean_tolerance_db,
            "plausible_std_min": args.plausible_std_min,
            "plausible_std_max": args.plausible_std_max,
            "max_patches_per_product": args.max_patches_per_product,
        },
        "n_scale_rows": len(scale_rows),
        "n_global_rows": len(global_rows),
        "n_city_rows": len(city_rows),
        "n_ssl4eo_rows": len(ssl4eo_rows),
        "outputs": {
            key: "" if value is None else path_to_str(value)
            for key, value in output_paths.items()
        },
    }


def summarize_manifest_count_rows(scale_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    # Placeholder for payload compatibility. The actual manifest row counts are logged and reflected
    # in the product-level processing. This returns zero because scale rows do not store counts.
    return [{"manifest_rows": 0}]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute patch-based SNAP-GRD vs RTC S1 VV/VH statistics and compare with SSL4EO-S12."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/croma_comparison_manifest_<stem>.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/s1_statistics_patch_based.",
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
        "--scale-probe-max-patches",
        type=int,
        default=256,
        help="Number of patch rows per product used for scale detection. Default: 256.",
    )

    parser.add_argument(
        "--scale-probe-pixels-per-patch",
        type=int,
        default=8192,
        help="Pixels sampled per patch for scale detection. Default: 8192.",
    )

    parser.add_argument(
        "--percentile-sample-per-window",
        type=int,
        default=2048,
        help="Pixels sampled per patch window for percentile/histogram estimates. Default: 2048.",
    )

    parser.add_argument(
        "--percentile-max-samples",
        type=int,
        default=1000000,
        help="Maximum retained samples per product/band for percentiles. Default: 1,000,000.",
    )

    parser.add_argument(
        "--treat-zero-as-invalid",
        action="store_true",
        help="Treat raw zero values as invalid. Default: false.",
    )

    parser.add_argument(
        "--plausible-mean-tolerance-db",
        type=float,
        default=10.0,
        help="Broad mean-difference tolerance against SSL4EO-S12. Default: 10 dB.",
    )

    parser.add_argument(
        "--plausible-std-min",
        type=float,
        default=1.0,
        help="Minimum plausible std in dB. Default: 1.",
    )

    parser.add_argument(
        "--plausible-std-max",
        type=float,
        default=12.0,
        help="Maximum plausible std in dB. Default: 12.",
    )

    parser.add_argument(
        "--max-patches-per-product",
        type=int,
        default=0,
        help="Debug option. If >0, use only first N manifest rows per product. Default: 0.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    parser.add_argument(
        "--make-figures",
        action="store_true",
        help="Generate histogram and city-mean figures if matplotlib is available.",
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
    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    manifest_path: Path = args.manifest_path or default_manifest_path(instance_root, stem)

    output_dir: Path = args.output_dir or (
        instance_root
        / "metadata"
        / "croma_probing"
        / "s1_statistics_patch_based"
    )

    scale_csv = output_dir / f"s1_patch_based_scale_detection_{stem}.csv"
    global_csv = output_dir / f"s1_patch_based_global_statistics_{stem}.csv"
    city_csv = output_dir / f"s1_patch_based_city_statistics_{stem}.csv"
    ssl4eo_csv = output_dir / f"s1_patch_based_ssl4eo_comparison_{stem}.csv"
    json_path = output_dir / f"s1_patch_based_statistics_summary_{stem}.json"
    md_path = output_dir / f"s1_patch_based_statistics_summary_{stem}.md"

    figure_hist_vv: Optional[Path] = None
    figure_hist_vh: Optional[Path] = None
    figure_city_vv: Optional[Path] = None
    figure_city_vh: Optional[Path] = None

    if args.make_figures:
        figure_dir = output_dir / "figures"
        figure_hist_vv = figure_dir / f"s1_patch_based_histogram_vv_{stem}.png"
        figure_hist_vh = figure_dir / f"s1_patch_based_histogram_vh_{stem}.png"
        figure_city_vv = figure_dir / f"s1_patch_based_city_mean_vv_{stem}.png"
        figure_city_vh = figure_dir / f"s1_patch_based_city_mean_vh_{stem}.png"

    output_paths: Dict[str, Optional[Path]] = {
        "scale_detection_csv": scale_csv,
        "global_statistics_csv": global_csv,
        "city_statistics_csv": city_csv,
        "ssl4eo_comparison_csv": ssl4eo_csv,
        "json": json_path,
        "markdown": md_path,
        "histogram_vv": figure_hist_vv,
        "histogram_vh": figure_hist_vh,
        "city_mean_vv": figure_city_vv,
        "city_mean_vh": figure_city_vh,
    }

    log("STEP", "Computing patch-based S1 SNAP-GRD vs RTC statistics.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Manifest:      {path_to_str(manifest_path)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")
    log("INFO", f"Stem:          {stem}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not manifest_path.exists():
        fail(f"CROMA comparison manifest does not exist: {path_to_str(manifest_path)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    product_rows, columns = load_product_manifest_rows(manifest_path)

    manifest_rows_used = sum(len(rows) for rows in product_rows.values())

    scale_rows, scale_info = detect_scales(
        product_rows,
        columns,
        patch_size=int(args.patch_size),
        scale_probe_max_patches=int(args.scale_probe_max_patches),
        scale_probe_pixels_per_patch=int(args.scale_probe_pixels_per_patch),
        treat_zero_as_invalid=bool(args.treat_zero_as_invalid),
        random_state=int(args.random_state),
    )

    global_rows, city_rows, sample_values = compute_patch_based_statistics(
        product_rows,
        columns,
        scale_info,
        patch_size=int(args.patch_size),
        percentile_sample_per_window=int(args.percentile_sample_per_window),
        percentile_max_samples=int(args.percentile_max_samples),
        treat_zero_as_invalid=bool(args.treat_zero_as_invalid),
        random_state=int(args.random_state),
        max_patches_per_product=int(args.max_patches_per_product),
    )

    ssl4eo_rows = build_ssl4eo_comparison_rows(
        global_rows,
        plausible_mean_tolerance_db=float(args.plausible_mean_tolerance_db),
        plausible_std_min=float(args.plausible_std_min),
        plausible_std_max=float(args.plausible_std_max),
    )

    main_conclusion = build_main_conclusion(scale_rows, ssl4eo_rows)

    if args.make_figures:
        output_paths["histogram_vv"] = make_histogram_figure(
            sample_values=sample_values,
            band="VV",
            output_path=figure_hist_vv,
            overwrite=bool(args.overwrite),
        )

        output_paths["histogram_vh"] = make_histogram_figure(
            sample_values=sample_values,
            band="VH",
            output_path=figure_hist_vh,
            overwrite=bool(args.overwrite),
        )

        output_paths["city_mean_vv"] = make_city_mean_figure(
            city_rows=city_rows,
            band="VV",
            output_path=figure_city_vv,
            overwrite=bool(args.overwrite),
        )

        output_paths["city_mean_vh"] = make_city_mean_figure(
            city_rows=city_rows,
            band="VH",
            output_path=figure_city_vh,
            overwrite=bool(args.overwrite),
        )

    summary_payload = {
        "created_utc": now_utc(),
        "status": "passed",
        "instance_root": path_to_str(instance_root),
        "manifest_path": path_to_str(manifest_path),
        "output_dir": path_to_str(output_dir),
        "patch_size": int(args.patch_size),
        "stride": int(args.stride),
        "edge_mode": str(args.edge_mode),
        "total_manifest_rows_used": int(manifest_rows_used),
        "main_conclusion": main_conclusion,
        "ssl4eo_reference": SSL4EO_S1_REFERENCE,
        "parameters": {
            "scale_probe_max_patches": int(args.scale_probe_max_patches),
            "scale_probe_pixels_per_patch": int(args.scale_probe_pixels_per_patch),
            "percentile_sample_per_window": int(args.percentile_sample_per_window),
            "percentile_max_samples": int(args.percentile_max_samples),
            "treat_zero_as_invalid": bool(args.treat_zero_as_invalid),
            "plausible_mean_tolerance_db": float(args.plausible_mean_tolerance_db),
            "plausible_std_min": float(args.plausible_std_min),
            "plausible_std_max": float(args.plausible_std_max),
            "max_patches_per_product": int(args.max_patches_per_product),
        },
        "n_scale_rows": len(scale_rows),
        "n_global_rows": len(global_rows),
        "n_city_rows": len(city_rows),
        "n_ssl4eo_rows": len(ssl4eo_rows),
        "outputs": {
            key: "" if value is None else path_to_str(value)
            for key, value in output_paths.items()
        },
    }

    log("STEP", "Writing S1 statistics outputs.")

    write_csv(
        scale_csv,
        scale_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "product",
            "band",
            "raw_sample_count",
            "raw_min",
            "raw_p01",
            "raw_p05",
            "raw_median",
            "raw_mean",
            "raw_p95",
            "raw_p99",
            "raw_max",
            "raw_std",
            "inferred_scale",
            "convert_to_db",
            "notes",
        ],
    )

    write_csv(
        global_csv,
        global_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "product",
            "band",
            "scale_used",
            "inferred_input_scale",
            "converted_to_db",
            "valid_pixel_count",
            "total_pixel_count",
            "invalid_pixel_count",
            "invalid_percent",
            "mean",
            "median",
            "std",
            "min",
            "p01",
            "p05",
            "p25",
            "p75",
            "p95",
            "p99",
            "max",
            "percentile_sample_count",
        ],
    )

    write_csv(
        city_csv,
        city_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "product",
            "band",
            "city",
            "scale_used",
            "inferred_input_scale",
            "converted_to_db",
            "valid_pixel_count",
            "total_pixel_count",
            "invalid_pixel_count",
            "invalid_percent",
            "mean",
            "std",
            "min",
            "max",
        ],
    )

    write_csv(
        ssl4eo_csv,
        ssl4eo_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "product",
            "band",
            "our_mean",
            "ssl4eo_mean",
            "mean_difference",
            "our_std",
            "ssl4eo_std",
            "std_difference",
            "std_ratio_our_over_ssl4eo",
            "plausible_mean_tolerance_db",
            "plausible_std_min",
            "plausible_std_max",
            "plausibility_flag",
            "interpretation",
        ],
    )

    write_json(json_path, summary_payload, overwrite=bool(args.overwrite))

    write_markdown(
        md_path,
        summary=summary_payload,
        scale_rows=scale_rows,
        global_rows=global_rows,
        ssl4eo_rows=ssl4eo_rows,
        output_paths=output_paths,
        overwrite=bool(args.overwrite),
    )

    log("OK", f"Wrote scale CSV:    {path_to_str(scale_csv)}")
    log("OK", f"Wrote global CSV:   {path_to_str(global_csv)}")
    log("OK", f"Wrote city CSV:     {path_to_str(city_csv)}")
    log("OK", f"Wrote SSL4EO CSV:   {path_to_str(ssl4eo_csv)}")
    log("OK", f"Wrote JSON:         {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:     {path_to_str(md_path)}")

    for key in ["histogram_vv", "histogram_vh", "city_mean_vv", "city_mean_vh"]:
        if output_paths.get(key) is not None:
            log("OK", f"Wrote figure {key}: {path_to_str(output_paths[key])}")

    log("STEP", "Final S1 statistics summary.")
    log("OK", "Status: passed")
    log("OK", f"Manifest rows used: {manifest_rows_used}")
    log("OK", f"Main conclusion: {main_conclusion}")


if __name__ == "__main__":
    main()