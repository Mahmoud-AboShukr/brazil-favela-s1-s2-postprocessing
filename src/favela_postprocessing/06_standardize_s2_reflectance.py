#!/usr/bin/env python3
"""
Standardize finalized Sentinel-2 city composites to float32 reflectance.

Input:
    <output_root>/s2_final/<city>/<city>_s2_allbands_10m.tif

Output:
    <output_root>/dataset_instances/instance_B_standard_rs/s2/<city>/
        <city>_s2_12bands_reflectance_10m.tif

QC:
    <output_root>/qc/s2_reflectance_standardization_qc.csv

Example:
    python src/favela_postprocessing/06_standardize_s2_reflectance.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling
from tqdm import tqdm


SCRIPT_NAME = "06_standardize_s2_reflectance.py"
INSTANCE_NAME = "instance_B_standard_rs"
DEFAULT_SCALE_THRESHOLD = 2.0
DEFAULT_SCALE_DIVISOR = 10000.0
DEFAULT_MAX_SAMPLE_PIXELS_PER_BAND = 250_000


@dataclass
class RunningStats:
    """Streaming statistics for one raster band."""

    total_pixels: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    min_value: float = math.nan
    max_value: float = math.nan
    sum_value: float = 0.0
    sumsq_value: float = 0.0
    zero_count: int = 0
    negative_count: int = 0
    gt_one_count: int = 0

    def update(self, arr: np.ndarray, valid_mask: np.ndarray) -> None:
        self.total_pixels += int(arr.size)
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
        self.zero_count += int(np.count_nonzero(values == 0))
        self.negative_count += int(np.count_nonzero(values < 0))
        self.gt_one_count += int(np.count_nonzero(values > 1))

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
        if self.total_pixels == 0:
            return math.nan
        return 100.0 * self.valid_count / self.total_pixels

    @property
    def invalid_percent(self) -> float:
        if self.total_pixels == 0:
            return math.nan
        return 100.0 * self.invalid_count / self.total_pixels

    @property
    def zero_percent_of_valid(self) -> float:
        if self.valid_count == 0:
            return math.nan
        return 100.0 * self.zero_count / self.valid_count

    @property
    def negative_percent_of_valid(self) -> float:
        if self.valid_count == 0:
            return math.nan
        return 100.0 * self.negative_count / self.valid_count

    @property
    def gt_one_percent_of_valid(self) -> float:
        if self.valid_count == 0:
            return math.nan
        return 100.0 * self.gt_one_count / self.valid_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize finalized Sentinel-2 products to float32 reflectance."
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
        help="Process only one city. Can be repeated, e.g. --city belem --city rio_de_janeiro",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["auto", "divide_10000", "keep"],
        default="auto",
        help="auto detects whether division by 10000 is needed. Default: auto",
    )
    parser.add_argument(
        "--scale-threshold",
        type=float,
        default=DEFAULT_SCALE_THRESHOLD,
        help="In auto mode, sampled p98 > threshold means divide by 10000. Default: 2.0",
    )
    parser.add_argument(
        "--max-sample-pixels-per-band",
        type=int,
        default=DEFAULT_MAX_SAMPLE_PIXELS_PER_BAND,
        help="Maximum approximate sample pixels per band for percentile estimation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files, overriding config overwrite:false.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned processing without writing outputs.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "output_root" not in cfg:
        raise KeyError("The config file must contain output_root.")

    return cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def valid_mask_for(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        mask &= arr != nodata
    return mask


def safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.percentile(values.astype("float32", copy=False), q))


def discover_s2_products(
    s2_final_root: Path,
    selected_cities: Optional[Sequence[str]],
) -> List[Tuple[str, Path]]:
    if not s2_final_root.exists():
        raise FileNotFoundError(f"S2 finalized root not found: {s2_final_root}")

    if selected_cities:
        cities = sorted(set(selected_cities))
    else:
        cities = sorted(path.name for path in s2_final_root.iterdir() if path.is_dir())

    products: List[Tuple[str, Path]] = []
    missing: List[str] = []

    for city in cities:
        expected_path = s2_final_root / city / f"{city}_s2_allbands_10m.tif"
        if expected_path.exists():
            products.append((city, expected_path))
            continue

        fallback_matches = sorted((s2_final_root / city).glob("*_s2_allbands_10m.tif"))
        if fallback_matches:
            products.append((city, fallback_matches[0]))
        else:
            missing.append(city)

    if missing:
        print("[WARN] Missing finalized S2 products for these cities:")
        for city in missing:
            print(f"       - {city}")

    if not products:
        raise RuntimeError(f"No finalized S2 products found under: {s2_final_root}")

    return products


def sample_valid_values_by_band(
    src: rasterio.io.DatasetReader,
    max_sample_pixels_per_band: int,
) -> List[np.ndarray]:
    total_pixels = src.width * src.height
    scale = max(1, int(math.ceil(math.sqrt(total_pixels / max_sample_pixels_per_band))))
    out_height = max(1, int(math.ceil(src.height / scale)))
    out_width = max(1, int(math.ceil(src.width / scale)))

    samples: List[np.ndarray] = []

    for band_index in range(1, src.count + 1):
        arr = src.read(
            band_index,
            out_shape=(out_height, out_width),
            resampling=Resampling.nearest,
            masked=False,
        ).astype("float32", copy=False)

        mask = valid_mask_for(arr, src.nodata)
        samples.append(arr[mask].copy())

    return samples


def decide_scale_factor(
    samples_by_band: Sequence[np.ndarray],
    scale_mode: str,
    scale_threshold: float,
) -> Tuple[float, str, str, float, float]:
    non_empty_samples = [sample for sample in samples_by_band if sample.size > 0]

    if not non_empty_samples:
        if scale_mode == "divide_10000":
            return (
                1.0 / DEFAULT_SCALE_DIVISOR,
                "divide_10000",
                "forced_no_valid_sample",
                math.nan,
                math.nan,
            )
        return 1.0, "keep", "no_valid_sample", math.nan, math.nan

    global_sample = np.concatenate(non_empty_samples)
    global_p98 = safe_percentile(global_sample, 98)
    global_max = float(np.max(global_sample))

    if scale_mode == "divide_10000":
        return 1.0 / DEFAULT_SCALE_DIVISOR, "divide_10000", "forced", global_p98, global_max

    if scale_mode == "keep":
        return 1.0, "keep", "forced", global_p98, global_max

    if np.isfinite(global_p98) and global_p98 > scale_threshold:
        return (
            1.0 / DEFAULT_SCALE_DIVISOR,
            "divide_10000",
            f"auto_global_sample_p98_{global_p98:.6g}_gt_{scale_threshold}",
            global_p98,
            global_max,
        )

    return (
        1.0,
        "keep",
        f"auto_global_sample_p98_{global_p98:.6g}_le_{scale_threshold}",
        global_p98,
        global_max,
    )


def choose_output_nodata(input_nodata: Optional[float], scale_factor: float) -> Optional[float]:
    if input_nodata is None:
        return None

    if not np.isfinite(input_nodata):
        return float("nan")

    if float(input_nodata) == 0.0:
        return 0.0

    if scale_factor != 1.0:
        return -9999.0

    return float(input_nodata)


def make_output_profile(
    src: rasterio.io.DatasetReader,
    compress: bool,
    output_nodata: Optional[float],
) -> Dict[str, Any]:
    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=src.count,
        nodata=output_nodata,
        BIGTIFF="IF_SAFER",
    )

    profile.pop("photometric", None)

    if compress:
        profile.update(compress="DEFLATE", predictor=3)
    else:
        profile.pop("compress", None)
        profile.pop("predictor", None)

    if src.width >= 512 and src.height >= 512:
        profile.update(tiled=True, blockxsize=512, blockysize=512)

    return profile


def make_temp_path(final_path: Path) -> Path:
    ensure_dir(final_path.parent)

    handle = tempfile.NamedTemporaryFile(
        prefix=f".{final_path.stem}_",
        suffix=".tmp.tif",
        dir=final_path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()

    return temp_path


def output_path_for_city(output_s2_root: Path, city: str) -> Path:
    return output_s2_root / city / f"{city}_s2_12bands_reflectance_10m.tif"


def set_output_metadata(
    dst: rasterio.io.DatasetWriter,
    src: rasterio.io.DatasetReader,
    city: str,
    input_path: Path,
    scale_mode_used: str,
    scale_factor: float,
    scale_reason: str,
) -> None:
    for band_index, description in enumerate(src.descriptions, start=1):
        if description:
            dst.set_band_description(band_index, description)

    dst.update_tags(
        city=city,
        source_file=str(input_path),
        processing_script=SCRIPT_NAME,
        dataset_instance=INSTANCE_NAME,
        scale_mode_used=scale_mode_used,
        scale_factor=str(scale_factor),
        scale_reason=scale_reason,
        output_units="surface_reflectance_float32",
    )


def sample_percentile_rows(
    samples_by_band: Sequence[np.ndarray],
    scale_factor: float,
) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}

    for band_index, sample in enumerate(samples_by_band, start=1):
        if sample.size == 0:
            stats[band_index] = {
                "input_p2_sample": math.nan,
                "input_p98_sample": math.nan,
                "output_p2_sample": math.nan,
                "output_p98_sample": math.nan,
            }
            continue

        output_sample = sample.astype("float32", copy=False) * np.float32(scale_factor)

        stats[band_index] = {
            "input_p2_sample": safe_percentile(sample, 2),
            "input_p98_sample": safe_percentile(sample, 98),
            "output_p2_sample": safe_percentile(output_sample, 2),
            "output_p98_sample": safe_percentile(output_sample, 98),
        }

    return stats


def process_city(
    city: str,
    input_path: Path,
    output_path: Path,
    compress: bool,
    overwrite: bool,
    dry_run: bool,
    scale_mode: str,
    scale_threshold: float,
    max_sample_pixels_per_band: int,
) -> List[Dict[str, Any]]:
    if output_path.exists() and not overwrite:
        print(f"[SKIP] {city}: output exists and overwrite is false")
        return [
            {
                "city": city,
                "status": "SKIPPED_EXISTS",
                "input_path": str(input_path),
                "output_path": str(output_path),
            }
        ]

    if dry_run:
        print(f"[DRY-RUN] {city}: {input_path} -> {output_path}")
        return [
            {
                "city": city,
                "status": "DRY_RUN",
                "input_path": str(input_path),
                "output_path": str(output_path),
            }
        ]

    temp_path: Optional[Path] = None

    try:
        with rasterio.open(input_path) as src:
            if src.count != 12:
                print(f"[WARN] {city}: expected 12 S2 bands, found {src.count}")

            samples_by_band = sample_valid_values_by_band(
                src=src,
                max_sample_pixels_per_band=max_sample_pixels_per_band,
            )

            (
                scale_factor,
                scale_mode_used,
                scale_reason,
                global_sample_p98,
                global_sample_max,
            ) = decide_scale_factor(
                samples_by_band=samples_by_band,
                scale_mode=scale_mode,
                scale_threshold=scale_threshold,
            )

            output_nodata = choose_output_nodata(src.nodata, scale_factor)
            sample_stats = sample_percentile_rows(samples_by_band, scale_factor)

            input_stats = [RunningStats() for _ in range(src.count)]
            output_stats = [RunningStats() for _ in range(src.count)]

            profile = make_output_profile(
                src=src,
                compress=compress,
                output_nodata=output_nodata,
            )

            temp_path = make_temp_path(output_path)

            with rasterio.open(temp_path, "w", **profile) as dst:
                set_output_metadata(
                    dst=dst,
                    src=src,
                    city=city,
                    input_path=input_path,
                    scale_mode_used=scale_mode_used,
                    scale_factor=scale_factor,
                    scale_reason=scale_reason,
                )

                for _, window in src.block_windows(1):
                    block = src.read(window=window, masked=False).astype("float32", copy=False)
                    output_block = np.empty(block.shape, dtype="float32")

                    for band_zero_index in range(src.count):
                        band_values = block[band_zero_index]

                        input_valid = valid_mask_for(band_values, src.nodata)
                        input_stats[band_zero_index].update(band_values, input_valid)

                        output_band = band_values.astype("float32", copy=True)
                        output_band[input_valid] = output_band[input_valid] * np.float32(scale_factor)

                        if output_nodata is not None:
                            output_band[~input_valid] = np.float32(output_nodata)

                        output_valid = valid_mask_for(output_band, output_nodata)
                        output_stats[band_zero_index].update(output_band, output_valid)

                        output_block[band_zero_index] = output_band

                    dst.write(output_block, window=window)

            temp_path.replace(output_path)

            rows: List[Dict[str, Any]] = []

            for band_index in range(1, src.count + 1):
                in_stats = input_stats[band_index - 1]
                out_stats = output_stats[band_index - 1]
                percentiles = sample_stats[band_index]
                band_description = src.descriptions[band_index - 1] if src.descriptions else None

                rows.append(
                    {
                        "city": city,
                        "status": "OK",
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "band": band_index,
                        "band_description": band_description,
                        "input_dtype": src.dtypes[band_index - 1],
                        "output_dtype": "float32",
                        "input_nodata": src.nodata,
                        "output_nodata": output_nodata,
                        "band_count": src.count,
                        "width": src.width,
                        "height": src.height,
                        "crs": str(src.crs),
                        "transform": str(src.transform),
                        "resolution_x": abs(float(src.transform.a)),
                        "resolution_y": abs(float(src.transform.e)),
                        "scale_mode_used": scale_mode_used,
                        "scale_factor": scale_factor,
                        "scale_reason": scale_reason,
                        "global_sample_p98": global_sample_p98,
                        "global_sample_max": global_sample_max,
                        "input_valid_pixel_count": in_stats.valid_count,
                        "input_invalid_pixel_count": in_stats.invalid_count,
                        "input_valid_percent": in_stats.valid_percent,
                        "input_invalid_percent": in_stats.invalid_percent,
                        "input_min": in_stats.min_value,
                        "input_max": in_stats.max_value,
                        "input_mean": in_stats.mean,
                        "input_std": in_stats.std,
                        "input_p2_sample": percentiles["input_p2_sample"],
                        "input_p98_sample": percentiles["input_p98_sample"],
                        "output_valid_pixel_count": out_stats.valid_count,
                        "output_invalid_pixel_count": out_stats.invalid_count,
                        "output_valid_percent": out_stats.valid_percent,
                        "output_invalid_percent": out_stats.invalid_percent,
                        "output_min": out_stats.min_value,
                        "output_max": out_stats.max_value,
                        "output_mean": out_stats.mean,
                        "output_std": out_stats.std,
                        "output_p2_sample": percentiles["output_p2_sample"],
                        "output_p98_sample": percentiles["output_p98_sample"],
                        "output_zero_percent_of_valid": out_stats.zero_percent_of_valid,
                        "output_negative_percent_of_valid": out_stats.negative_percent_of_valid,
                        "output_gt_one_percent_of_valid": out_stats.gt_one_percent_of_valid,
                    }
                )

            print(
                f"[OK] {city}: {scale_mode_used}, "
                f"factor={scale_factor}, reason={scale_reason}"
            )

            return rows

    except Exception as exc:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

        print(f"[FAILED] {city}: {exc}")

        return [
            {
                "city": city,
                "status": "FAILED",
                "input_path": str(input_path),
                "output_path": str(output_path),
                "error": repr(exc),
            }
        ]


def write_qc(rows: List[Dict[str, Any]], qc_path: Path) -> None:
    ensure_dir(qc_path.parent)

    df = pd.DataFrame(rows)
    df.to_csv(qc_path, index=False)

    print(f"[INFO] Wrote QC CSV: {qc_path}")

    if "status" in df.columns:
        print("[INFO] Status counts:")
        print(df["status"].value_counts(dropna=False).to_string())

    if "status" in df.columns and "city" in df.columns:
        ok_cities = df.loc[df["status"] == "OK", "city"].nunique()
        if ok_cities > 0:
            print(f"[INFO] Cities successfully standardized: {ok_cities}")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))
    s2_final_root = Path(str(cfg.get("s2_final_root", output_root / "s2_final")))

    instance_root = Path(
        str(cfg.get("instance_b_root", output_root / "dataset_instances" / INSTANCE_NAME))
    )

    output_s2_root = Path(str(cfg.get("instance_b_s2_root", instance_root / "s2")))

    qc_path = Path(
        str(
            cfg.get(
                "s2_reflectance_qc_csv",
                output_root / "qc" / "s2_reflectance_standardization_qc.csv",
            )
        )
    )

    compress = bool(cfg.get("compress", True))
    overwrite = bool(cfg.get("overwrite", False)) or bool(args.overwrite)

    print("[INFO] Sentinel-2 reflectance standardization")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Input S2 root: {s2_final_root}")
    print(f"[INFO] Output S2 root: {output_s2_root}")
    print(f"[INFO] QC CSV: {qc_path}")
    print(f"[INFO] Compress: {compress}")
    print(f"[INFO] Overwrite: {overwrite}")
    print(f"[INFO] Scale mode: {args.scale_mode}")

    products = discover_s2_products(s2_final_root, args.city)

    print(f"[INFO] Finalized S2 city products discovered: {len(products)}")

    all_rows: List[Dict[str, Any]] = []

    for city, input_path in tqdm(products, desc="Standardizing S2 cities"):
        output_path = output_path_for_city(output_s2_root, city)

        rows = process_city(
            city=city,
            input_path=input_path,
            output_path=output_path,
            compress=compress,
            overwrite=overwrite,
            dry_run=args.dry_run,
            scale_mode=args.scale_mode,
            scale_threshold=args.scale_threshold,
            max_sample_pixels_per_band=args.max_sample_pixels_per_band,
        )

        all_rows.extend(rows)

    write_qc(all_rows, qc_path)

    statuses = {str(row.get("status")) for row in all_rows}

    if "FAILED" in statuses:
        print("[ERROR] Some cities failed. Check the QC CSV for details.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())