#!/usr/bin/env python3
"""
Compare direct-GCP Sentinel-1 GRD baseline against SNAP terrain-corrected S1 pilot outputs.

Purpose
-------
This script compares:

Instance A / current baseline:
    <output_root>/s1_final/<city>/<city>_s1_grd_vv_vh_vvdiff_10m_aligned.tif

against:

Instance B / SNAP pilot:
    <output_root>/dataset_instances/instance_B_standard_rs/s1_snap/<city>/
        <city>_s1_snap_vv_vh_vvdiff_10m_aligned.tif

The goal is not to prove that one is "better" automatically. The goal is to produce
a clean quantitative QC report before deciding whether to scale SNAP preprocessing to all cities.

The script checks:
    - file existence
    - CRS / transform / width / height / band count
    - nodata consistency
    - per-band direct-GCP statistics
    - per-band SNAP statistics
    - paired valid pixel percentage
    - direct minus SNAP difference statistics
    - Pearson correlation between direct and SNAP
    - mean absolute difference
    - RMSE

Optionally, it can write difference rasters:
    direct_GCP - SNAP

Outputs
-------
CSV:
    <output_root>/qc/s1_direct_vs_snap_pilot_comparison.csv

Markdown summary:
    <output_root>/qc/s1_direct_vs_snap_pilot_summary.md

Optional difference rasters:
    <output_root>/qc/rasters/s1_direct_vs_snap_pilot/<city>/
        <city>_s1_direct_minus_snap_vv_vh_vvdiff.tif

Examples
--------
Pilot comparison:

    python src/favela_postprocessing/08_compare_s1_direct_gcp_vs_snap_pilot.py --config configs/default.yaml

With difference rasters:

    python src/favela_postprocessing/08_compare_s1_direct_gcp_vs_snap_pilot.py --config configs/default.yaml --write-diff-rasters

One city only:

    python src/favela_postprocessing/08_compare_s1_direct_gcp_vs_snap_pilot.py --config configs/default.yaml --city rio_de_janeiro
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
import yaml
from tqdm import tqdm


SCRIPT_NAME = "08_compare_s1_direct_gcp_vs_snap_pilot.py"
INSTANCE_NAME = "instance_B_standard_rs"

PILOT_CITIES = [
    "rio_de_janeiro",
    "belem",
    "porto_alegre",
]

BAND_NAMES = {
    1: "VV_dB",
    2: "VH_dB",
    3: "VV_minus_VH_dB",
}

OUTPUT_NODATA = -9999.0


@dataclass
class StreamingStats:
    """Streaming univariate statistics."""

    total_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    min_value: float = math.nan
    max_value: float = math.nan
    sum_value: float = 0.0
    sumsq_value: float = 0.0
    sample_values: Optional[List[np.ndarray]] = None

    def __post_init__(self) -> None:
        if self.sample_values is None:
            self.sample_values = []

    def update(
        self,
        arr: np.ndarray,
        valid_mask: np.ndarray,
        keep_sample: bool = True,
        max_sample_per_update: int = 50_000,
    ) -> None:
        self.total_count += int(arr.size)

        valid_count = int(valid_mask.sum())
        self.valid_count += valid_count
        self.invalid_count += int(arr.size - valid_count)

        if valid_count == 0:
            return

        values = arr[valid_mask].astype("float64", copy=False)

        current_min = float(np.min(values))
        current_max = float(np.max(values))

        if math.isnan(self.min_value):
            self.min_value = current_min
        else:
            self.min_value = min(self.min_value, current_min)

        if math.isnan(self.max_value):
            self.max_value = current_max
        else:
            self.max_value = max(self.max_value, current_max)

        self.sum_value += float(np.sum(values, dtype="float64"))
        self.sumsq_value += float(np.sum(values * values, dtype="float64"))

        if keep_sample and self.sample_values is not None:
            if values.size > max_sample_per_update:
                step = max(1, values.size // max_sample_per_update)
                sampled = values[::step][:max_sample_per_update].astype("float32")
            else:
                sampled = values.astype("float32")

            self.sample_values.append(sampled)

    @property
    def mean(self) -> float:
        if self.valid_count == 0:
            return math.nan
        return self.sum_value / self.valid_count

    @property
    def std(self) -> float:
        if self.valid_count == 0:
            return math.nan
        variance = (self.sumsq_value / self.valid_count) - (self.mean * self.mean)
        return math.sqrt(max(variance, 0.0))

    @property
    def valid_percent(self) -> float:
        if self.total_count == 0:
            return math.nan
        return 100.0 * self.valid_count / self.total_count

    @property
    def invalid_percent(self) -> float:
        if self.total_count == 0:
            return math.nan
        return 100.0 * self.invalid_count / self.total_count

    def percentile(self, q: float) -> float:
        if not self.sample_values:
            return math.nan

        values = np.concatenate(self.sample_values)

        if values.size == 0:
            return math.nan

        return float(np.percentile(values, q))


@dataclass
class StreamingPairedStats:
    """Streaming paired statistics for direct vs SNAP."""

    paired_count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0
    sum_diff: float = 0.0
    sum_abs_diff: float = 0.0
    sumsq_diff: float = 0.0
    diff_stats: StreamingStats = None  # type: ignore

    def __post_init__(self) -> None:
        if self.diff_stats is None:
            self.diff_stats = StreamingStats()

    def update(self, x: np.ndarray, y: np.ndarray, paired_mask: np.ndarray) -> None:
        count = int(paired_mask.sum())

        if count == 0:
            return

        x_values = x[paired_mask].astype("float64", copy=False)
        y_values = y[paired_mask].astype("float64", copy=False)
        diff = x_values - y_values

        self.paired_count += count
        self.sum_x += float(np.sum(x_values, dtype="float64"))
        self.sum_y += float(np.sum(y_values, dtype="float64"))
        self.sum_x2 += float(np.sum(x_values * x_values, dtype="float64"))
        self.sum_y2 += float(np.sum(y_values * y_values, dtype="float64"))
        self.sum_xy += float(np.sum(x_values * y_values, dtype="float64"))
        self.sum_diff += float(np.sum(diff, dtype="float64"))
        self.sum_abs_diff += float(np.sum(np.abs(diff), dtype="float64"))
        self.sumsq_diff += float(np.sum(diff * diff, dtype="float64"))

        self.diff_stats.update(
            arr=diff.astype("float32"),
            valid_mask=np.ones(diff.shape, dtype=bool),
            keep_sample=True,
        )

    @property
    def mean_diff(self) -> float:
        if self.paired_count == 0:
            return math.nan
        return self.sum_diff / self.paired_count

    @property
    def mean_abs_diff(self) -> float:
        if self.paired_count == 0:
            return math.nan
        return self.sum_abs_diff / self.paired_count

    @property
    def rmse(self) -> float:
        if self.paired_count == 0:
            return math.nan
        return math.sqrt(self.sumsq_diff / self.paired_count)

    @property
    def pearson_r(self) -> float:
        n = self.paired_count

        if n <= 1:
            return math.nan

        numerator = (n * self.sum_xy) - (self.sum_x * self.sum_y)
        denom_x = (n * self.sum_x2) - (self.sum_x * self.sum_x)
        denom_y = (n * self.sum_y2) - (self.sum_y * self.sum_y)

        denominator = math.sqrt(max(denom_x, 0.0) * max(denom_y, 0.0))

        if denominator == 0:
            return math.nan

        return numerator / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct-GCP S1 baseline against SNAP terrain-corrected S1 pilot products."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="Compare only one city. Can be repeated.",
    )
    parser.add_argument(
        "--write-diff-rasters",
        action="store_true",
        help="Write direct-GCP minus SNAP difference rasters.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing difference rasters.",
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


def get_cities(selected: Optional[Sequence[str]]) -> List[str]:
    if selected:
        return sorted(set(normalize_city_name(city) for city in selected))

    return PILOT_CITIES.copy()


def direct_s1_path(output_root: Path, city: str) -> Path:
    return output_root / "s1_final" / city / f"{city}_s1_grd_vv_vh_vvdiff_10m_aligned.tif"


def snap_s1_path(output_root: Path, city: str) -> Path:
    return (
        output_root
        / "dataset_instances"
        / INSTANCE_NAME
        / "s1_snap"
        / city
        / f"{city}_s1_snap_vv_vh_vvdiff_10m_aligned.tif"
    )


def diff_raster_path(output_root: Path, city: str) -> Path:
    return (
        output_root
        / "qc"
        / "rasters"
        / "s1_direct_vs_snap_pilot"
        / city
        / f"{city}_s1_direct_minus_snap_vv_vh_vvdiff.tif"
    )


def valid_mask(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    mask = np.isfinite(arr)

    if nodata is not None and np.isfinite(nodata):
        mask &= arr != nodata

    return mask


def transforms_equal(a: Any, b: Any, tolerance: float = 1e-9) -> bool:
    try:
        return all(abs(x - y) <= tolerance for x, y in zip(a, b))
    except Exception:
        return False


def compare_grid(direct: rasterio.io.DatasetReader, snap: rasterio.io.DatasetReader) -> Dict[str, Any]:
    same_crs = direct.crs == snap.crs
    same_transform = transforms_equal(direct.transform, snap.transform)
    same_width = direct.width == snap.width
    same_height = direct.height == snap.height
    same_count = direct.count == snap.count

    return {
        "same_crs": same_crs,
        "same_transform": same_transform,
        "same_width": same_width,
        "same_height": same_height,
        "same_shape": same_width and same_height,
        "same_band_count": same_count,
        "direct_crs": str(direct.crs),
        "snap_crs": str(snap.crs),
        "direct_transform": str(direct.transform),
        "snap_transform": str(snap.transform),
        "direct_width": direct.width,
        "direct_height": direct.height,
        "snap_width": snap.width,
        "snap_height": snap.height,
        "direct_band_count": direct.count,
        "snap_band_count": snap.count,
        "direct_nodata": direct.nodata,
        "snap_nodata": snap.nodata,
    }


def make_diff_profile(reference: rasterio.io.DatasetReader, compress: bool = True) -> Dict[str, Any]:
    profile = reference.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=3,
        nodata=OUTPUT_NODATA,
        BIGTIFF="IF_SAFER",
    )

    profile.pop("photometric", None)

    if compress:
        profile.update(compress="DEFLATE", predictor=3)

    if reference.width >= 512 and reference.height >= 512:
        profile.update(tiled=True, blockxsize=512, blockysize=512)

    return profile


def compare_city(
    output_root: Path,
    city: str,
    write_diff_rasters: bool,
    overwrite: bool,
) -> List[Dict[str, Any]]:
    direct_path = direct_s1_path(output_root, city)
    snap_path = snap_s1_path(output_root, city)
    out_diff_path = diff_raster_path(output_root, city)

    base = {
        "city": city,
        "direct_path": str(direct_path),
        "snap_path": str(snap_path),
        "diff_raster_path": str(out_diff_path) if write_diff_rasters else "",
    }

    if not direct_path.exists():
        row = dict(base)
        row.update({"status": "FAILED_MISSING_DIRECT_GCP"})
        return [row]

    if not snap_path.exists():
        row = dict(base)
        row.update({"status": "FAILED_MISSING_SNAP"})
        return [row]

    rows: List[Dict[str, Any]] = []

    with rasterio.open(direct_path) as direct, rasterio.open(snap_path) as snap:
        grid = compare_grid(direct, snap)

        can_compare_pixelwise = (
            grid["same_crs"]
            and grid["same_transform"]
            and grid["same_shape"]
            and grid["same_band_count"]
            and direct.count >= 3
            and snap.count >= 3
        )

        if not can_compare_pixelwise:
            row = dict(base)
            row.update(grid)
            row.update({"status": "FAILED_GRID_MISMATCH"})
            return [row]

        direct_stats = {band: StreamingStats() for band in range(1, 4)}
        snap_stats = {band: StreamingStats() for band in range(1, 4)}
        paired_stats = {band: StreamingPairedStats() for band in range(1, 4)}

        diff_writer = None

        try:
            if write_diff_rasters:
                if out_diff_path.exists() and not overwrite:
                    print(f"[WARN] Difference raster exists and overwrite is false: {out_diff_path}")
                    print("[WARN] Difference raster will not be rewritten.")
                else:
                    ensure_dir(out_diff_path.parent)
                    profile = make_diff_profile(direct, compress=True)
                    diff_writer = rasterio.open(out_diff_path, "w", **profile)
                    diff_writer.set_band_description(1, "direct_minus_snap_VV_dB")
                    diff_writer.set_band_description(2, "direct_minus_snap_VH_dB")
                    diff_writer.set_band_description(3, "direct_minus_snap_VV_minus_VH_dB")
                    diff_writer.update_tags(
                        city=city,
                        processing_script=SCRIPT_NAME,
                        direct_source=str(direct_path),
                        snap_source=str(snap_path),
                        units="dB_difference",
                    )

            for _, window in direct.block_windows(1):
                direct_block = direct.read(indexes=[1, 2, 3], window=window).astype("float32")
                snap_block = snap.read(indexes=[1, 2, 3], window=window).astype("float32")

                diff_block = np.full(direct_block.shape, OUTPUT_NODATA, dtype="float32")

                for band_zero, band in enumerate([1, 2, 3]):
                    direct_arr = direct_block[band_zero]
                    snap_arr = snap_block[band_zero]

                    direct_valid = valid_mask(direct_arr, direct.nodata)
                    snap_valid = valid_mask(snap_arr, snap.nodata)
                    paired_valid = direct_valid & snap_valid

                    direct_stats[band].update(direct_arr, direct_valid)
                    snap_stats[band].update(snap_arr, snap_valid)
                    paired_stats[band].update(direct_arr, snap_arr, paired_valid)

                    diff_arr = diff_block[band_zero]
                    diff_arr[paired_valid] = direct_arr[paired_valid] - snap_arr[paired_valid]

                if diff_writer is not None:
                    diff_writer.write(diff_block, window=window)

        finally:
            if diff_writer is not None:
                diff_writer.close()

        for band in range(1, 4):
            direct_stat = direct_stats[band]
            snap_stat = snap_stats[band]
            paired_stat = paired_stats[band]

            total_pixels = direct.width * direct.height
            paired_valid_percent = (
                100.0 * paired_stat.paired_count / total_pixels if total_pixels > 0 else math.nan
            )

            row = dict(base)
            row.update(grid)
            row.update(
                {
                    "status": "OK",
                    "band": band,
                    "band_name": BAND_NAMES.get(band, f"band_{band}"),
                    "total_pixels": total_pixels,
                    "paired_valid_count": paired_stat.paired_count,
                    "paired_valid_percent": paired_valid_percent,
                    "direct_valid_count": direct_stat.valid_count,
                    "direct_valid_percent": direct_stat.valid_percent,
                    "direct_min": direct_stat.min_value,
                    "direct_max": direct_stat.max_value,
                    "direct_mean": direct_stat.mean,
                    "direct_std": direct_stat.std,
                    "direct_p2": direct_stat.percentile(2),
                    "direct_p98": direct_stat.percentile(98),
                    "snap_valid_count": snap_stat.valid_count,
                    "snap_valid_percent": snap_stat.valid_percent,
                    "snap_min": snap_stat.min_value,
                    "snap_max": snap_stat.max_value,
                    "snap_mean": snap_stat.mean,
                    "snap_std": snap_stat.std,
                    "snap_p2": snap_stat.percentile(2),
                    "snap_p98": snap_stat.percentile(98),
                    "diff_direct_minus_snap_min": paired_stat.diff_stats.min_value,
                    "diff_direct_minus_snap_max": paired_stat.diff_stats.max_value,
                    "diff_direct_minus_snap_mean": paired_stat.mean_diff,
                    "diff_direct_minus_snap_std": paired_stat.diff_stats.std,
                    "diff_direct_minus_snap_p2": paired_stat.diff_stats.percentile(2),
                    "diff_direct_minus_snap_p98": paired_stat.diff_stats.percentile(98),
                    "mean_abs_difference": paired_stat.mean_abs_diff,
                    "rmse": paired_stat.rmse,
                    "pearson_r": paired_stat.pearson_r,
                    "diff_raster_written": bool(write_diff_rasters and out_diff_path.exists()),
                }
            )

            rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> pd.DataFrame:
    ensure_dir(csv_path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def markdown_escape(value: Any) -> str:
    """Convert a value to Markdown-safe text without requiring tabulate."""
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


def df_to_markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    """
    Minimal Markdown table writer.

    This avoids pandas.to_markdown(), which requires the optional 'tabulate'
    dependency.
    """
    if df.empty:
        return "_No rows._"

    table_df = df.copy()

    if index:
        table_df = table_df.reset_index()

    columns = list(table_df.columns)

    header = "| " + " | ".join(markdown_escape(col) for col in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    rows = []

    for _, row in table_df.iterrows():
        rows.append(
            "| "
            + " | ".join(markdown_escape(row[col]) for col in columns)
            + " |"
        )

    return "\n".join([header, separator] + rows)


def series_to_markdown_table(series: pd.Series, index_name: str, value_name: str) -> str:
    """Convert a Series into a two-column Markdown table without tabulate."""
    if series.empty:
        return "_No rows._"

    table_df = (
        series
        .rename_axis(index_name)
        .reset_index(name=value_name)
    )

    return df_to_markdown_table(table_df, index=False)


def write_markdown_summary(df: pd.DataFrame, md_path: Path) -> None:
    ensure_dir(md_path.parent)

    lines: List[str] = []

    lines.append("# Sentinel-1 Direct-GCP vs SNAP Pilot Comparison")
    lines.append("")
    lines.append(f"Generated by `{SCRIPT_NAME}`.")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report compares the current direct-GCP Sentinel-1 GRD baseline "
        "against the SNAP terrain-corrected pilot outputs for the selected cities."
    )
    lines.append("")
    lines.append(
        "The comparison is a QC/diagnostic report. It does not by itself prove which product is better."
    )
    lines.append("")

    if df.empty:
        lines.append("No rows were generated.")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Status counts")
    lines.append("")
    lines.append(
        series_to_markdown_table(
            df["status"].value_counts(dropna=False),
            index_name="status",
            value_name="row_count",
        )
    )
    lines.append("")

    ok = df[df["status"] == "OK"].copy()

    if ok.empty:
        lines.append("No successful comparisons were produced.")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return

    city_count = ok["city"].nunique()

    lines.append("## Successful comparison summary")
    lines.append("")
    lines.append(f"- Cities compared successfully: **{city_count}**")
    lines.append(f"- Bands compared per city: **{ok['band'].nunique()}**")
    lines.append("")

    lines.append("## Grid alignment")
    lines.append("")

    grid_cols = [
        "city",
        "same_crs",
        "same_transform",
        "same_shape",
        "same_band_count",
        "direct_crs",
        "snap_crs",
        "direct_width",
        "direct_height",
    ]

    existing_grid_cols = [col for col in grid_cols if col in ok.columns]
    grid_df = ok[existing_grid_cols].drop_duplicates(subset=["city"])

    lines.append(df_to_markdown_table(grid_df, index=False))
    lines.append("")

    lines.append("## Per-band comparison")
    lines.append("")

    compact_cols = [
        "city",
        "band_name",
        "paired_valid_percent",
        "direct_mean",
        "snap_mean",
        "diff_direct_minus_snap_mean",
        "mean_abs_difference",
        "rmse",
        "pearson_r",
        "direct_p2",
        "direct_p98",
        "snap_p2",
        "snap_p98",
    ]

    existing_compact_cols = [col for col in compact_cols if col in ok.columns]
    compact = ok[existing_compact_cols].copy()

    for col in compact.columns:
        if col not in ["city", "band_name"]:
            compact[col] = compact[col].map(lambda x: fmt_float(x, 3))

    lines.append(df_to_markdown_table(compact, index=False))
    lines.append("")

    lines.append("## How to interpret")
    lines.append("")
    lines.append("- `paired_valid_percent` close to 100% means both products cover the same pixels.")
    lines.append("- `pearson_r` close to 1 means both products have similar spatial patterns.")
    lines.append("- `mean_abs_difference` and `rmse` quantify how far the dB values are numerically.")
    lines.append(
        "- SNAP and direct-GCP outputs can differ because SNAP applies orbit correction, "
        "thermal-noise removal, calibration, and terrain correction."
    )
    lines.append(
        "- A large numeric difference is not automatically bad; it should be interpreted "
        "with visual QC and downstream segmentation tests."
    )
    lines.append("")

    lines.append("## Recommended next decision")
    lines.append("")
    lines.append(
        "If SNAP products show better visual/geometric consistency and reasonable distributions, "
        "scale SAFE download + SNAP preprocessing to all cities. If direct-GCP and SNAP look nearly "
        "equivalent, consider keeping the simpler direct-GCP baseline and documenting the pilot comparison."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))
    cities = get_cities(args.city)

    csv_path = output_root / "qc" / "s1_direct_vs_snap_pilot_comparison.csv"
    md_path = output_root / "qc" / "s1_direct_vs_snap_pilot_summary.md"

    print("[INFO] Sentinel-1 direct-GCP vs SNAP pilot comparison")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Cities: {', '.join(cities)}")
    print(f"[INFO] Write difference rasters: {args.write_diff_rasters}")
    print(f"[INFO] CSV output: {csv_path}")
    print(f"[INFO] Markdown output: {md_path}")

    all_rows: List[Dict[str, Any]] = []

    for city in tqdm(cities, desc="Comparing S1 products"):
        rows = compare_city(
            output_root=output_root,
            city=city,
            write_diff_rasters=args.write_diff_rasters,
            overwrite=args.overwrite,
        )

        all_rows.extend(rows)

        statuses = sorted(set(str(row.get("status")) for row in rows))
        print(f"[INFO] {city}: {', '.join(statuses)}")

    df = write_csv(all_rows, csv_path)
    write_markdown_summary(df, md_path)

    print(f"[INFO] Wrote CSV: {csv_path}")
    print(f"[INFO] Wrote Markdown summary: {md_path}")

    if not df.empty and "status" in df.columns:
        print("[INFO] Status counts:")
        print(df["status"].value_counts(dropna=False).to_string())

    if not df.empty and (df["status"] != "OK").any():
        print("[ERROR] Some comparisons failed. Check the CSV.")
        return 1

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())