#!/usr/bin/env python3
"""
Compute training-only normalization statistics for ML experiments.

Purpose
-------
This script computes normalization statistics from TRAIN patches only.

It reads one patch filter-set CSV produced by:

    14_build_patch_filter_sets.py

and computes per-band/channel statistics for:

    - Sentinel-2 reflectance bands
    - SNAP Sentinel-1 channels

It does NOT use validation or test patches to compute statistics.

Why this matters
----------------
Normalization must be computed from the training split only. If validation/test pixels
are used to compute mean/std/percentiles, then information from the evaluation set leaks
into the training pipeline.

Default input
-------------
By default, this script uses:

    <output_root>/metadata/patch_filter_sets_train_covered_region_test_ps512_st512_cover/
        filter_set_F02_quality_pass.csv

Outputs
-------
JSON:
    <output_root>/metadata/normalization_stats_train_covered_region_test_ps512_st512_cover_F02_quality_pass.json

CSV:
    <output_root>/metadata/normalization_stats_train_covered_region_test_ps512_st512_cover_F02_quality_pass.csv

Markdown:
    <output_root>/metadata/normalization_stats_train_covered_region_test_ps512_st512_cover_F02_quality_pass.md

Example
-------
    python3 src/favela_postprocessing/15_compute_normalization_stats.py --config configs/default.yaml

Use another filter set:

    python3 src/favela_postprocessing/15_compute_normalization_stats.py \
        --config configs/default.yaml \
        --filter-set-id F09_strict_nodata5_pos_plus_neg_1_2_train_eval_all
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import yaml
from tqdm import tqdm


SCRIPT_NAME = "15_compute_normalization_stats.py"

DEFAULT_SPLIT_STRATEGY = "train_covered_region_test"
DEFAULT_PATCH_SIZE = 512
DEFAULT_STRIDE = 512
DEFAULT_EDGE_MODE = "cover"
DEFAULT_FILTER_SET_ID = "F02_quality_pass"

S2_BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]

S1_CHANNEL_NAMES = [
    "VV_dB",
    "VH_dB",
    "VV_minus_VH_dB",
]


@dataclass
class StreamingBandStats:
    """Streaming statistics for one raster band/channel."""

    name: str
    valid_count: int = 0
    invalid_count: int = 0
    min_value: float = math.nan
    max_value: float = math.nan
    sum_value: float = 0.0
    sumsq_value: float = 0.0
    samples: List[np.ndarray] = field(default_factory=list)

    def update(
        self,
        values: np.ndarray,
        max_samples_per_update: int,
    ) -> None:
        if values.size == 0:
            return

        values = values.astype("float64", copy=False)

        self.valid_count += int(values.size)
        self.sum_value += float(np.sum(values, dtype="float64"))
        self.sumsq_value += float(np.sum(values * values, dtype="float64"))

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

        if max_samples_per_update > 0:
            if values.size > max_samples_per_update:
                step = max(1, values.size // max_samples_per_update)
                sample = values[::step][:max_samples_per_update].astype("float32")
            else:
                sample = values.astype("float32")

            self.samples.append(sample)

    def add_invalid(self, count: int) -> None:
        self.invalid_count += int(count)

    @property
    def total_count(self) -> int:
        return self.valid_count + self.invalid_count

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

    def percentile(self, q: float) -> float:
        if not self.samples:
            return math.nan

        values = np.concatenate(self.samples)

        if values.size == 0:
            return math.nan

        return float(np.percentile(values, q))

    def to_row(
        self,
        modality: str,
        band_index: int,
        percentile_note: str,
    ) -> Dict[str, Any]:
        return {
            "modality": modality,
            "band_index": band_index,
            "band_name": self.name,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "total_count": self.total_count,
            "valid_percent": self.valid_percent,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.mean,
            "std": self.std,
            "p1": self.percentile(1),
            "p2": self.percentile(2),
            "p5": self.percentile(5),
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p98": self.percentile(98),
            "p99": self.percentile(99),
            "percentile_note": percentile_note,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute training-only normalization statistics for S2 and SNAP S1 patches."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--filter-set",
        type=Path,
        default=None,
        help="Explicit filter-set CSV path. If omitted, uses default lookup from filter-set-id.",
    )
    parser.add_argument(
        "--filter-set-id",
        type=str,
        default=DEFAULT_FILTER_SET_ID,
        help=f"Filter set ID to use for default lookup. Default: {DEFAULT_FILTER_SET_ID}",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default=DEFAULT_SPLIT_STRATEGY,
        help=f"Split strategy suffix. Default: {DEFAULT_SPLIT_STRATEGY}",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Patch size suffix. Default: {DEFAULT_PATCH_SIZE}",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Stride suffix. Default: {DEFAULT_STRIDE}",
    )
    parser.add_argument(
        "--edge-mode",
        type=str,
        default=DEFAULT_EDGE_MODE,
        help=f"Edge mode suffix. Default: {DEFAULT_EDGE_MODE}",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split used to compute normalization. Default: train",
    )
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="Compute stats for only one city. Can be repeated. Usually not used.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Optional patch limit for testing.",
    )
    parser.add_argument(
        "--max-samples-per-patch-band",
        type=int,
        default=5000,
        help=(
            "Maximum pixel samples stored per patch per band for approximate percentiles. "
            "Mean/std are exact streaming values. Default: 5000"
        ),
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


def default_filter_set_path(
    output_root: Path,
    split_strategy: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
    filter_set_id: str,
) -> Path:
    suffix = f"{split_strategy}_ps{patch_size}_st{stride}_{edge_mode}"
    return (
        output_root
        / "metadata"
        / f"patch_filter_sets_{suffix}"
        / f"filter_set_{filter_set_id}.csv"
    )


def suffix_from_filter_set(path: Path) -> str:
    parent_name = path.parent.name
    prefix = "patch_filter_sets_"

    if parent_name.startswith(prefix):
        tiling_suffix = parent_name[len(prefix):]
    else:
        tiling_suffix = parent_name

    file_name = path.stem
    filter_prefix = "filter_set_"

    if file_name.startswith(filter_prefix):
        filter_set_id = file_name[len(filter_prefix):]
    else:
        filter_set_id = file_name

    return f"{tiling_suffix}_{filter_set_id}"


def load_filter_set(
    path: Path,
    split_name: str,
    selected_cities: Optional[Sequence[str]],
    max_patches: Optional[int],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Filter-set CSV not found: {path}\n"
            f"Run 14_build_patch_filter_sets.py first."
        )

    df = pd.read_csv(path)

    required = [
        "patch_id",
        "city",
        "split",
        "source_s2_path",
        "source_s1_path",
        "row_start",
        "row_end",
        "col_start",
        "col_end",
        "patch_height",
        "patch_width",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Filter set is missing required columns: {missing}")

    df = df.copy()
    df["city"] = df["city"].map(normalize_city_name)
    df["split"] = df["split"].astype(str)

    df = df[df["split"] == split_name].copy()

    if selected_cities:
        selected = {normalize_city_name(city) for city in selected_cities}
        df = df[df["city"].isin(selected)].copy()

    if max_patches is not None:
        df = df.head(max_patches).copy()

    if df.empty:
        raise RuntimeError(f"No patches selected for split='{split_name}'.")

    return df.reset_index(drop=True)


def window_from_row(row: pd.Series) -> Window:
    return Window(
        col_off=int(row["col_start"]),
        row_off=int(row["row_start"]),
        width=int(row["patch_width"]),
        height=int(row["patch_height"]),
    )


def valid_mask_for_array(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
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


def init_stats(names: Sequence[str]) -> List[StreamingBandStats]:
    return [
        StreamingBandStats(name=name)
        for name in names
    ]


def update_s2_stats(
    s2_stats: List[StreamingBandStats],
    s2_path: Path,
    window: Window,
    treat_all_zero_as_nodata: bool,
    max_samples_per_patch_band: int,
) -> None:
    with rasterio.open(s2_path) as src:
        arr = src.read(window=window).astype("float32", copy=False)

        if arr.shape[0] != len(s2_stats):
            raise RuntimeError(
                f"Unexpected S2 band count for {s2_path}: "
                f"got {arr.shape[0]}, expected {len(s2_stats)}"
            )

        pixel_valid = valid_mask_for_array(arr, src.nodata)

        if treat_all_zero_as_nodata:
            pixel_valid = pixel_valid & ~all_zero_mask(arr)

        total_pixels = int(arr.shape[1] * arr.shape[2])

        for band_idx, stat in enumerate(s2_stats):
            band = arr[band_idx]
            values = band[pixel_valid]
            stat.update(values, max_samples_per_patch_band)
            stat.add_invalid(total_pixels - int(values.size))


def update_s1_stats(
    s1_stats: List[StreamingBandStats],
    s1_path: Path,
    window: Window,
    max_samples_per_patch_band: int,
) -> None:
    with rasterio.open(s1_path) as src:
        arr = src.read(window=window).astype("float32", copy=False)

        if arr.shape[0] != len(s1_stats):
            raise RuntimeError(
                f"Unexpected S1 band count for {s1_path}: "
                f"got {arr.shape[0]}, expected {len(s1_stats)}"
            )

        pixel_valid = valid_mask_for_array(arr, src.nodata)
        total_pixels = int(arr.shape[1] * arr.shape[2])

        for band_idx, stat in enumerate(s1_stats):
            band = arr[band_idx]
            values = band[pixel_valid]
            stat.update(values, max_samples_per_patch_band)
            stat.add_invalid(total_pixels - int(values.size))


def rows_from_stats(
    s2_stats: List[StreamingBandStats],
    s1_stats: List[StreamingBandStats],
    percentile_note: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for idx, stat in enumerate(s2_stats, start=1):
        rows.append(
            stat.to_row(
                modality="s2_reflectance",
                band_index=idx,
                percentile_note=percentile_note,
            )
        )

    for idx, stat in enumerate(s1_stats, start=1):
        rows.append(
            stat.to_row(
                modality="s1_snap",
                band_index=idx,
                percentile_note=percentile_note,
            )
        )

    return pd.DataFrame(rows)


def make_json_output(
    stats_df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "metadata": metadata,
        "s2_reflectance": {},
        "s1_snap": {},
    }

    for _, row in stats_df.iterrows():
        modality = str(row["modality"])
        band_name = str(row["band_name"])

        item = {
            "band_index": int(row["band_index"]),
            "valid_count": int(row["valid_count"]),
            "invalid_count": int(row["invalid_count"]),
            "total_count": int(row["total_count"]),
            "valid_percent": float(row["valid_percent"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "p1": float(row["p1"]),
            "p2": float(row["p2"]),
            "p5": float(row["p5"]),
            "p50": float(row["p50"]),
            "p95": float(row["p95"]),
            "p98": float(row["p98"]),
            "p99": float(row["p99"]),
        }

        output[modality][band_name] = item

    return output


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
    md_path: Path,
    stats_df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> None:
    ensure_dir(md_path.parent)

    lines: List[str] = []

    lines.append("# Training-Only Normalization Statistics")
    lines.append("")
    lines.append(f"Generated by `{SCRIPT_NAME}`.")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "This file reports normalization statistics computed from training patches only. "
        "Validation and test patches are intentionally excluded to avoid data leakage."
    )
    lines.append("")

    lines.append("## Metadata")
    lines.append("")

    metadata_df = pd.DataFrame(
        [
            {"key": key, "value": value}
            for key, value in metadata.items()
        ]
    )

    lines.append(df_to_markdown_table(metadata_df))
    lines.append("")

    lines.append("## Recommended normalization values")
    lines.append("")

    display_cols = [
        "modality",
        "band_index",
        "band_name",
        "mean",
        "std",
        "p2",
        "p98",
        "valid_percent",
    ]

    display = stats_df[display_cols].copy()

    for col in ["mean", "std", "p2", "p98", "valid_percent"]:
        display[col] = display[col].map(lambda x: f"{float(x):.6f}")

    lines.append(df_to_markdown_table(display))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `mean` and `std` are exact streaming statistics over selected training pixels.")
    lines.append("- Percentiles are estimated from sampled pixels to avoid storing all raster values in memory.")
    lines.append("- For Sentinel-2, all-zero pixels can be treated as nodata if enabled.")
    lines.append("- These statistics should be applied unchanged to train, validation, and test patches.")
    lines.append("")

    lines.append("## Recommended next step")
    lines.append("")
    lines.append(
        "Use this JSON/CSV in the PyTorch GeoTIFF dataloader so that all model inputs "
        "are normalized consistently using training-only statistics."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))
    metadata_root = output_root / "metadata"
    ensure_dir(metadata_root)

    if args.filter_set is None:
        filter_set_path = default_filter_set_path(
            output_root=output_root,
            split_strategy=args.split_strategy,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            filter_set_id=args.filter_set_id,
        )
    else:
        filter_set_path = args.filter_set

    suffix = suffix_from_filter_set(filter_set_path)

    json_path = metadata_root / f"normalization_stats_{suffix}.json"
    csv_path = metadata_root / f"normalization_stats_{suffix}.csv"
    md_path = metadata_root / f"normalization_stats_{suffix}.md"

    print("[INFO] Compute training-only normalization statistics")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Filter set: {filter_set_path}")
    print(f"[INFO] Split used for stats: {args.split}")
    print(f"[INFO] Output JSON: {json_path}")
    print(f"[INFO] Output CSV: {csv_path}")
    print(f"[INFO] Output Markdown: {md_path}")
    print(f"[INFO] Treat all-zero S2 as nodata: {args.treat_all_zero_s2_as_nodata}")
    print(f"[INFO] Max samples per patch/band for percentiles: {args.max_samples_per_patch_band}")

    df = load_filter_set(
        path=filter_set_path,
        split_name=args.split,
        selected_cities=args.city,
        max_patches=args.max_patches,
    )

    print(f"[INFO] Training patches selected: {len(df)}")
    print(f"[INFO] Cities represented: {df['city'].nunique()}")
    print(f"[INFO] Regions represented: {df['region'].nunique() if 'region' in df.columns else 'UNKNOWN'}")

    s2_stats = init_stats(S2_BAND_NAMES)
    s1_stats = init_stats(S1_CHANNEL_NAMES)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading training patches"):
        window = window_from_row(row)

        update_s2_stats(
            s2_stats=s2_stats,
            s2_path=Path(str(row["source_s2_path"])),
            window=window,
            treat_all_zero_as_nodata=args.treat_all_zero_s2_as_nodata,
            max_samples_per_patch_band=args.max_samples_per_patch_band,
        )

        update_s1_stats(
            s1_stats=s1_stats,
            s1_path=Path(str(row["source_s1_path"])),
            window=window,
            max_samples_per_patch_band=args.max_samples_per_patch_band,
        )

    percentile_note = (
        "Percentiles are approximate, estimated from sampled pixels. "
        f"Up to {args.max_samples_per_patch_band} pixels were sampled per patch per band."
    )

    stats_df = rows_from_stats(
        s2_stats=s2_stats,
        s1_stats=s1_stats,
        percentile_note=percentile_note,
    )

    stats_df.to_csv(csv_path, index=False)

    metadata = {
        "script": SCRIPT_NAME,
        "output_root": str(output_root),
        "filter_set_path": str(filter_set_path),
        "filter_set_suffix": suffix,
        "split_used_for_statistics": args.split,
        "patch_count_used": int(len(df)),
        "city_count_used": int(df["city"].nunique()),
        "region_count_used": int(df["region"].nunique()) if "region" in df.columns else None,
        "treat_all_zero_s2_as_nodata": bool(args.treat_all_zero_s2_as_nodata),
        "max_samples_per_patch_band": int(args.max_samples_per_patch_band),
        "important_note": (
            "These statistics were computed from training patches only. "
            "Do not recompute them using validation or test data."
        ),
    }

    json_output = make_json_output(stats_df, metadata)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    write_markdown_summary(
        md_path=md_path,
        stats_df=stats_df,
        metadata=metadata,
    )

    print(f"[INFO] Wrote JSON: {json_path}")
    print(f"[INFO] Wrote CSV: {csv_path}")
    print(f"[INFO] Wrote Markdown: {md_path}")

    print("[INFO] Normalization statistics:")
    display = stats_df[
        [
            "modality",
            "band_index",
            "band_name",
            "mean",
            "std",
            "p2",
            "p98",
            "valid_percent",
        ]
    ].copy()

    print(display.to_string(index=False))

    print("[INFO] Training-only normalization statistics computed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())