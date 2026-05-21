#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
11_compute_s1_snap_rtc_statistics_224.py

Patch-based raw Sentinel-1 statistics for SNAP-GRD vs RTC, with optional
forced RTC conversion to dB.

Purpose
-------
Compute VV/VH pixel-value statistics for the exact 224x224 patch set used in
the CROMA comparison, then compare SNAP-GRD and RTC against the published
SSL4EO-S12 Sentinel-1 reference statistics.

This script supports two modes:

1. Diagnostic/default mode
   - Detect whether each product looks like dB or positive linear/power scale.
   - Convert only if the scale is clearly linear sigma0-like.
   - This is useful for diagnosing scaling problems.

2. Forced RTC conversion mode
   - Use --force-rtc-linear-to-db
   - Forces RTC VV/VH values through:
         dB = 10 * log10(value)
   - This makes RTC comparable with SNAP-GRD and SSL4EO-S12 on a dB scale.

Important
---------
This is NOT an embedding analysis and NOT a model-performance metric.
It directly inspects the SAR raster values used by the CROMA patch pipeline.

SSL4EO-S12 reference
--------------------
Published Sentinel-1 GRD statistics, dB scale:

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

For each patch row, it reads bands 1 and 2:

    band 1 = VV
    band 2 = VH

Because this is Option B, statistics are computed over the patch windows used
in the CROMA experiments. Since patch stride is 112 and patch size is 224,
overlapping pixels are counted multiple times. This is intentional: it describes
the distribution seen by the patch-based ML/CROMA pipeline.

Outputs
-------
<instance-root>/metadata/croma_probing/s1_statistics_patch_based/

    s1_patch_based_scale_detection_ps224_st112_cover.csv
    s1_patch_based_global_statistics_ps224_st112_cover.csv
    s1_patch_based_city_statistics_ps224_st112_cover.csv
    s1_patch_based_ssl4eo_comparison_ps224_st112_cover.csv
    s1_patch_based_outlier_diagnostics_ps224_st112_cover.csv
    s1_patch_based_statistics_summary_ps224_st112_cover.json
    s1_patch_based_statistics_summary_ps224_st112_cover.md

Optional figures:
    figures/s1_patch_based_histogram_vv_ps224_st112_cover.png
    figures/s1_patch_based_histogram_vh_ps224_st112_cover.png
    figures/s1_patch_based_city_mean_vv_ps224_st112_cover.png
    figures/s1_patch_based_city_mean_vh_ps224_st112_cover.png

Recommended forced-RTC run
--------------------------
python src/croma_probing/11_compute_s1_snap_rtc_statistics_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --force-rtc-linear-to-db `
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

PRODUCT_BAND_SEED_OFFSET = {
    ("SNAP-GRD", "VV"): 101,
    ("SNAP-GRD", "VH"): 102,
    ("RTC", "VV"): 201,
    ("RTC", "VH"): 202,
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


def product_band_seed(product: str, band: str, random_state: int) -> int:
    return int(random_state) + PRODUCT_BAND_SEED_OFFSET.get((product, band), 999)


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
    outlier_rows: List[Dict[str, object]],
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
    lines.append(f"- Forced RTC linear-to-dB conversion: `{summary['parameters']['force_rtc_linear_to_db']}`")
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

    lines.append("## Scale detection and conversion rule")
    lines.append("")
    lines.append("| product | band | inferred input scale | conversion mode | convert to dB | raw mean | raw median | raw p01 | raw p99 | transformed sample mean | transformed sample median | notes |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")

    for row in scale_rows:
        lines.append(
            f"| {row['product']} | "
            f"{row['band']} | "
            f"{row['inferred_input_scale']} | "
            f"{row['conversion_mode']} | "
            f"{row['convert_to_db']} | "
            f"{row['raw_mean']} | "
            f"{row['raw_median']} | "
            f"{row['raw_p01']} | "
            f"{row['raw_p99']} | "
            f"{row['transformed_sample_mean']} | "
            f"{row['transformed_sample_median']} | "
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
    lines.append("## Outlier diagnostics")
    lines.append("")
    lines.append("| product | band | raw > 1 % | raw > 10 % | raw > 100 % | transformed > 0 dB % | transformed > 10 dB % | transformed < -50 dB % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for row in outlier_rows:
        lines.append(
            f"| {row['product']} | "
            f"{row['band']} | "
            f"{row['raw_gt_1_percent']} | "
            f"{row['raw_gt_10_percent']} | "
            f"{row['raw_gt_100_percent']} | "
            f"{row['transformed_gt_0db_percent']} | "
            f"{row['transformed_gt_10db_percent']} | "
            f"{row['transformed_lt_minus50db_percent']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The key interpretation is whether the VV and VH values fall into a plausible Sentinel-1 dB range after the appropriate scale handling. SNAP-GRD is expected to already be in dB. RTC may be stored in a positive linear/power-like scale; therefore, when `--force-rtc-linear-to-db` is used, RTC is transformed with `10*log10(value)` before comparison.")
    lines.append("")
    lines.append("The comparison with SSL4EO-S12 is used only as a broad distribution sanity check. Exact agreement is not expected because SSL4EO-S12 is global and multi-seasonal, whereas this dataset is Brazil-urban/favela-focused.")
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
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Optional[str]]]:
    rows = read_csv_rows(manifest_path)
    fieldnames = list(rows[0].keys())

    columns: Dict[str, Optional[str]] = {
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
    assert modality_col is not None

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
    assert columns["row_off"] is not None
    assert columns["col_off"] is not None

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


class CounterStats:
    def __init__(self) -> None:
        self.total = 0
        self.raw_gt_1 = 0
        self.raw_gt_10 = 0
        self.raw_gt_100 = 0
        self.raw_gt_1000 = 0
        self.raw_le_0 = 0
        self.transformed_gt_0db = 0
        self.transformed_gt_10db = 0
        self.transformed_lt_minus50db = 0

    def update(self, raw_valid: np.ndarray, transformed_valid: np.ndarray) -> None:
        raw = np.asarray(raw_valid, dtype=np.float64)
        raw = raw[np.isfinite(raw)]

        transformed = np.asarray(transformed_valid, dtype=np.float64)
        transformed = transformed[np.isfinite(transformed)]

        self.total += int(raw.size)

        if raw.size > 0:
            self.raw_gt_1 += int(np.count_nonzero(raw > 1.0))
            self.raw_gt_10 += int(np.count_nonzero(raw > 10.0))
            self.raw_gt_100 += int(np.count_nonzero(raw > 100.0))
            self.raw_gt_1000 += int(np.count_nonzero(raw > 1000.0))
            self.raw_le_0 += int(np.count_nonzero(raw <= 0.0))

        if transformed.size > 0:
            self.transformed_gt_0db += int(np.count_nonzero(transformed > 0.0))
            self.transformed_gt_10db += int(np.count_nonzero(transformed > 10.0))
            self.transformed_lt_minus50db += int(np.count_nonzero(transformed < -50.0))

    def pct(self, count: int) -> float:
        if self.total <= 0:
            return 0.0
        return 100.0 * count / self.total

    def to_row(self, product: str, band: str) -> Dict[str, object]:
        return {
            "product": product,
            "band": band,
            "raw_valid_count": int(self.total),
            "raw_gt_1_count": int(self.raw_gt_1),
            "raw_gt_10_count": int(self.raw_gt_10),
            "raw_gt_100_count": int(self.raw_gt_100),
            "raw_gt_1000_count": int(self.raw_gt_1000),
            "raw_le_0_count": int(self.raw_le_0),
            "raw_gt_1_percent": round_float(self.pct(self.raw_gt_1), 8),
            "raw_gt_10_percent": round_float(self.pct(self.raw_gt_10), 8),
            "raw_gt_100_percent": round_float(self.pct(self.raw_gt_100), 8),
            "raw_gt_1000_percent": round_float(self.pct(self.raw_gt_1000), 8),
            "raw_le_0_percent": round_float(self.pct(self.raw_le_0), 8),
            "transformed_gt_0db_count": int(self.transformed_gt_0db),
            "transformed_gt_10db_count": int(self.transformed_gt_10db),
            "transformed_lt_minus50db_count": int(self.transformed_lt_minus50db),
            "transformed_gt_0db_percent": round_float(self.pct(self.transformed_gt_0db), 8),
            "transformed_gt_10db_percent": round_float(self.pct(self.transformed_gt_10db), 8),
            "transformed_lt_minus50db_percent": round_float(self.pct(self.transformed_lt_minus50db), 8),
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
    assert columns["sar_path"] is not None

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
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
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
        "p25": round_float(float(np.quantile(arr, 0.25)), 8),
        "median": round_float(float(np.median(arr)), 8),
        "p75": round_float(float(np.quantile(arr, 0.75)), 8),
        "mean": round_float(float(np.mean(arr)), 8),
        "p95": round_float(float(np.quantile(arr, 0.95)), 8),
        "p99": round_float(float(np.quantile(arr, 0.99)), 8),
        "max": round_float(float(np.max(arr)), 8),
        "std": round_float(float(np.std(arr)), 8),
    }


def infer_input_scale(values: np.ndarray) -> Tuple[str, str]:
    stats = summarize_array(values)

    p01 = safe_float(stats["p01"])
    p99 = safe_float(stats["p99"])
    median = safe_float(stats["median"])
    mean = safe_float(stats["mean"])
    min_value = safe_float(stats["min"])
    max_value = safe_float(stats["max"])

    if stats["count"] == 0:
        return "unknown_no_valid_values", "No valid values found during scale probe."

    if median < 0 and p01 > -100 and p99 < 50:
        return "db", "Values look like Sentinel-1 dB backscatter."

    if min_value >= 0 and p99 <= 5.0 and median < 1.5 and mean < 1.5:
        return "linear_sigma0_power", "Values look like linear sigma0 power."

    if min_value >= 0 and median < 1.5 and p99 > 5.0:
        return "positive_power_with_large_tail", "Values are positive with a small median but a very large upper tail."

    if min_value >= 0 and max_value > 5.0:
        return "unknown_positive_large", "Values are positive but not clearly standard linear sigma0."

    return "unknown_assumed_db", "Scale is ambiguous."


def determine_conversion(
    *,
    product: str,
    band: str,
    inferred_input_scale: str,
    force_rtc_linear_to_db: bool,
    force_all_positive_to_db: bool,
) -> Tuple[bool, str, str]:
    if force_rtc_linear_to_db and product == "RTC":
        return (
            True,
            "forced_rtc_10log10",
            "Forced RTC conversion to dB using 10*log10(value).",
        )

    if force_all_positive_to_db and inferred_input_scale in {
        "linear_sigma0_power",
        "positive_power_with_large_tail",
        "unknown_positive_large",
    }:
        return (
            True,
            "forced_positive_10log10",
            "Forced positive-valued input conversion to dB using 10*log10(value).",
        )

    if inferred_input_scale == "linear_sigma0_power":
        return (
            True,
            "auto_linear_sigma0_10log10",
            "Automatically converted linear sigma0-like values to dB using 10*log10(value).",
        )

    return (
        False,
        "none_raw_values_used",
        "No conversion applied.",
    )


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


def detect_scales(
    product_rows: Dict[str, List[Dict[str, str]]],
    columns: Dict[str, Optional[str]],
    *,
    patch_size: int,
    scale_probe_max_patches: int,
    scale_probe_pixels_per_patch: int,
    treat_zero_as_invalid: bool,
    random_state: int,
    force_rtc_linear_to_db: bool,
    force_all_positive_to_db: bool,
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
                raw_values = np.concatenate(band_samples[band_name])
            else:
                raw_values = np.asarray([], dtype=np.float32)

            inferred_input_scale, inference_note = infer_input_scale(raw_values)

            convert_to_db, conversion_mode, conversion_note = determine_conversion(
                product=product,
                band=band_name,
                inferred_input_scale=inferred_input_scale,
                force_rtc_linear_to_db=force_rtc_linear_to_db,
                force_all_positive_to_db=force_all_positive_to_db,
            )

            transformed_values, conversion_invalid_count = transform_values_to_comparison_scale(
                raw_values,
                convert_to_db=convert_to_db,
            )

            raw_stats = summarize_array(raw_values)
            transformed_stats = summarize_array(transformed_values)

            notes = f"{inference_note} {conversion_note}".strip()

            row = {
                "product": product,
                "band": band_name,

                "raw_sample_count": raw_stats["count"],
                "raw_min": raw_stats["min"],
                "raw_p01": raw_stats["p01"],
                "raw_p05": raw_stats["p05"],
                "raw_median": raw_stats["median"],
                "raw_mean": raw_stats["mean"],
                "raw_p95": raw_stats["p95"],
                "raw_p99": raw_stats["p99"],
                "raw_max": raw_stats["max"],
                "raw_std": raw_stats["std"],

                "inferred_input_scale": inferred_input_scale,
                "conversion_mode": conversion_mode,
                "convert_to_db": bool(convert_to_db),
                "conversion_invalid_count": int(conversion_invalid_count),

                "transformed_sample_count": transformed_stats["count"],
                "transformed_sample_min": transformed_stats["min"],
                "transformed_sample_p01": transformed_stats["p01"],
                "transformed_sample_p05": transformed_stats["p05"],
                "transformed_sample_median": transformed_stats["median"],
                "transformed_sample_mean": transformed_stats["mean"],
                "transformed_sample_p95": transformed_stats["p95"],
                "transformed_sample_p99": transformed_stats["p99"],
                "transformed_sample_max": transformed_stats["max"],
                "transformed_sample_std": transformed_stats["std"],

                "notes": notes,
            }

            scale_rows.append(row)

            scale_info[(product, band_name)] = {
                "inferred_input_scale": inferred_input_scale,
                "conversion_mode": conversion_mode,
                "convert_to_db": bool(convert_to_db),
                "notes": notes,
            }

            log(
                "OK",
                f"{product} {band_name}: input_scale={inferred_input_scale}, "
                f"conversion={conversion_mode}, "
                f"raw_mean={raw_stats['mean']}, "
                f"transformed_mean={transformed_stats['mean']}",
            )

    return scale_rows, scale_info


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
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[Tuple[str, str], np.ndarray],
]:
    global_stats: Dict[Tuple[str, str], RunningStats] = {}
    city_stats: Dict[Tuple[str, str, str], RunningStats] = {}
    counter_stats: Dict[Tuple[str, str], CounterStats] = {}
    sample_stores: Dict[Tuple[str, str], SampleStore] = {}

    for product in PRODUCT_MODALITIES:
        for band in ["VV", "VH"]:
            key = (product, band)
            global_stats[key] = RunningStats()
            counter_stats[key] = CounterStats()
            sample_stores[key] = SampleStore(
                max_samples=percentile_max_samples,
                random_state=product_band_seed(product, band, random_state),
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

            assert columns["city"] is not None
            city = str(row[columns["city"]]).strip()

            band_arrays = read_patch_bands(row, columns, patch_size=patch_size)

            for band_index, arr in band_arrays.items():
                band_name = BAND_INDEX_TO_NAME.get(band_index)

                if band_name is None:
                    continue

                raw_values_all = np.asarray(arr, dtype=np.float32).reshape(-1)
                raw_mask = np.isfinite(raw_values_all)

                if treat_zero_as_invalid:
                    raw_mask &= raw_values_all != 0

                raw_values = raw_values_all[raw_mask]

                convert_to_db = bool(scale_info[(product, band_name)]["convert_to_db"])

                transformed_values, conversion_invalid = transform_values_to_comparison_scale(
                    raw_values,
                    convert_to_db=convert_to_db,
                )

                total_count = int(arr.size)
                invalid_count = int(total_count - transformed_values.size)

                key = (product, band_name)

                global_stats[key].update_invalid_total(
                    total_count=total_count,
                    invalid_count=invalid_count,
                )
                global_stats[key].update(transformed_values)

                city_key = (product, band_name, city)

                if city_key not in city_stats:
                    city_stats[city_key] = RunningStats()

                city_stats[city_key].update_invalid_total(
                    total_count=total_count,
                    invalid_count=invalid_count,
                )
                city_stats[city_key].update(transformed_values)

                counter_stats[key].update(
                    raw_valid=raw_values,
                    transformed_valid=transformed_values,
                )

                sample_stores[key].add(
                    transformed_values,
                    per_window=percentile_sample_per_window,
                )

    sample_values: Dict[Tuple[str, str], np.ndarray] = {
        key: store.values()
        for key, store in sample_stores.items()
    }

    global_rows: List[Dict[str, object]] = []

    for product in PRODUCT_MODALITIES:
        for band in ["VV", "VH"]:
            key = (product, band)
            stats = global_stats[key].to_basic_dict()
            samples = sample_values[key]
            pct = summarize_array(samples)
            scale = scale_info[key]

            if scale["convert_to_db"]:
                scale_used = "converted_to_dB"
            elif scale["inferred_input_scale"] == "db":
                scale_used = "dB"
            else:
                scale_used = "raw_unconverted"

            global_rows.append(
                {
                    "product": product,
                    "band": band,
                    "scale_used": scale_used,
                    "inferred_input_scale": scale["inferred_input_scale"],
                    "conversion_mode": scale["conversion_mode"],
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
                    "p25": pct["p25"],
                    "p75": pct["p75"],
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

        if scale["convert_to_db"]:
            scale_used = "converted_to_dB"
        elif scale["inferred_input_scale"] == "db":
            scale_used = "dB"
        else:
            scale_used = "raw_unconverted"

        city_rows.append(
            {
                "product": product,
                "band": band,
                "city": city,
                "scale_used": scale_used,
                "inferred_input_scale": scale["inferred_input_scale"],
                "conversion_mode": scale["conversion_mode"],
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

    outlier_rows: List[Dict[str, object]] = []

    for product in PRODUCT_MODALITIES:
        for band in ["VV", "VH"]:
            outlier_rows.append(counter_stats[(product, band)].to_row(product, band))

    return global_rows, city_rows, outlier_rows, sample_values


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
        comparable_scale = str(row["scale_used"]) in {"dB", "converted_to_dB"}

        if not comparable_scale:
            flag = "not_comparable_scale"
        elif plausible_mean and plausible_std:
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
                "scale_used": row["scale_used"],
                "conversion_mode": row["conversion_mode"],
                "our_mean": round_float(our_mean, 8),
                "ssl4eo_mean": ref["mean"],
                "mean_difference": round_float(mean_difference, 8),
                "abs_mean_difference": round_float(abs(mean_difference), 8),
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
                    "exact agreement is not expected because domains differ."
                ),
            }
        )

    return rows


def build_main_conclusion(
    *,
    scale_rows: List[Dict[str, object]],
    ssl4eo_rows: List[Dict[str, object]],
    force_rtc_linear_to_db: bool,
) -> str:
    rtc_scale_rows = [row for row in scale_rows if row["product"] == "RTC"]
    rtc_converted = all(bool(row["convert_to_db"]) for row in rtc_scale_rows)

    snap_rows = [row for row in ssl4eo_rows if row["product"] == "SNAP-GRD"]
    rtc_rows = [row for row in ssl4eo_rows if row["product"] == "RTC"]

    snap_plausible = all(row["plausibility_flag"] == "broadly_plausible" for row in snap_rows)
    rtc_plausible = all(row["plausibility_flag"] == "broadly_plausible" for row in rtc_rows)

    if snap_plausible and rtc_plausible:
        return (
            "Both SNAP-GRD and RTC are broadly plausible relative to the SSL4EO-S12 "
            "Sentinel-1 dB reference statistics after the applied scale handling. "
            "This supports the physical plausibility of both S1 inputs for the patch-based CROMA pipeline."
        )

    if snap_plausible and force_rtc_linear_to_db and rtc_converted and not rtc_plausible:
        return (
            "SNAP-GRD is broadly plausible relative to SSL4EO-S12. RTC was forced into dB using "
            "10*log10(value), making it comparable in scale, but its mean and/or standard deviation "
            "still differs from the SSL4EO-S12 reference range. This suggests that RTC may have a "
            "different calibration/distribution, strong urban-domain effects, or remaining outlier influence. "
            "The RTC distribution should be inspected with the outlier and city-level diagnostics."
        )

    if snap_plausible and not force_rtc_linear_to_db:
        return (
            "SNAP-GRD is broadly plausible relative to SSL4EO-S12, while RTC is not directly comparable "
            "unless converted to dB. Run again with --force-rtc-linear-to-db to compare RTC with SNAP-GRD "
            "and SSL4EO-S12 on a common dB scale."
        )

    if snap_plausible:
        return (
            "SNAP-GRD is broadly plausible relative to SSL4EO-S12. RTC requires additional inspection "
            "because its distribution remains shifted after scale handling."
        )

    return (
        "The distributions require inspection before drawing a physical plausibility conclusion. "
        "Check scale detection, conversion mode, outlier diagnostics, and city-level statistics."
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
    ax.set_xlabel(f"{band} backscatter value, dB scale if converted/applied")
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
    ax.set_ylabel(f"Mean {band}, dB scale if converted/applied")
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=60, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


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
        "--force-rtc-linear-to-db",
        action="store_true",
        help="Force RTC VV/VH conversion to dB using 10*log10(value).",
    )

    parser.add_argument(
        "--force-all-positive-to-db",
        action="store_true",
        help="Force all positive-valued non-dB products to dB. Usually not needed; prefer --force-rtc-linear-to-db.",
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
    outlier_csv = output_dir / f"s1_patch_based_outlier_diagnostics_{stem}.csv"
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
        "outlier_diagnostics_csv": outlier_csv,
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
    log("INFO", f"Force RTC to dB: {bool(args.force_rtc_linear_to_db)}")

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
        force_rtc_linear_to_db=bool(args.force_rtc_linear_to_db),
        force_all_positive_to_db=bool(args.force_all_positive_to_db),
    )

    global_rows, city_rows, outlier_rows, sample_values = compute_patch_based_statistics(
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

    main_conclusion = build_main_conclusion(
        scale_rows=scale_rows,
        ssl4eo_rows=ssl4eo_rows,
        force_rtc_linear_to_db=bool(args.force_rtc_linear_to_db),
    )

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
            "force_rtc_linear_to_db": bool(args.force_rtc_linear_to_db),
            "force_all_positive_to_db": bool(args.force_all_positive_to_db),
            "plausible_mean_tolerance_db": float(args.plausible_mean_tolerance_db),
            "plausible_std_min": float(args.plausible_std_min),
            "plausible_std_max": float(args.plausible_std_max),
            "max_patches_per_product": int(args.max_patches_per_product),
        },
        "n_scale_rows": len(scale_rows),
        "n_global_rows": len(global_rows),
        "n_city_rows": len(city_rows),
        "n_ssl4eo_rows": len(ssl4eo_rows),
        "n_outlier_rows": len(outlier_rows),
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

            "inferred_input_scale",
            "conversion_mode",
            "convert_to_db",
            "conversion_invalid_count",

            "transformed_sample_count",
            "transformed_sample_min",
            "transformed_sample_p01",
            "transformed_sample_p05",
            "transformed_sample_median",
            "transformed_sample_mean",
            "transformed_sample_p95",
            "transformed_sample_p99",
            "transformed_sample_max",
            "transformed_sample_std",

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
            "conversion_mode",
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
            "conversion_mode",
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
            "scale_used",
            "conversion_mode",
            "our_mean",
            "ssl4eo_mean",
            "mean_difference",
            "abs_mean_difference",
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

    write_csv(
        outlier_csv,
        outlier_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "product",
            "band",
            "raw_valid_count",
            "raw_gt_1_count",
            "raw_gt_10_count",
            "raw_gt_100_count",
            "raw_gt_1000_count",
            "raw_le_0_count",
            "raw_gt_1_percent",
            "raw_gt_10_percent",
            "raw_gt_100_percent",
            "raw_gt_1000_percent",
            "raw_le_0_percent",
            "transformed_gt_0db_count",
            "transformed_gt_10db_count",
            "transformed_lt_minus50db_count",
            "transformed_gt_0db_percent",
            "transformed_gt_10db_percent",
            "transformed_lt_minus50db_percent",
        ],
    )

    write_json(json_path, summary_payload, overwrite=bool(args.overwrite))

    write_markdown(
        md_path,
        summary=summary_payload,
        scale_rows=scale_rows,
        global_rows=global_rows,
        ssl4eo_rows=ssl4eo_rows,
        outlier_rows=outlier_rows,
        output_paths=output_paths,
        overwrite=bool(args.overwrite),
    )

    log("OK", f"Wrote scale CSV:    {path_to_str(scale_csv)}")
    log("OK", f"Wrote global CSV:   {path_to_str(global_csv)}")
    log("OK", f"Wrote city CSV:     {path_to_str(city_csv)}")
    log("OK", f"Wrote SSL4EO CSV:   {path_to_str(ssl4eo_csv)}")
    log("OK", f"Wrote outlier CSV:  {path_to_str(outlier_csv)}")
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