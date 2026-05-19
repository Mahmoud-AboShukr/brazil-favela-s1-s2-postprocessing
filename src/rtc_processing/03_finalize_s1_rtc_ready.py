#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_finalize_s1_rtc_ready.py

Finalize Sentinel-1 RTC rasters into Instance C.

This script reads the georeferencing decision table created by:

    src/rtc_processing/02_inspect_rtc_georeferencing.py

Input decision table:

    <instance-root>/metadata/rtc_processing/rtc_georeferencing_by_city.csv

It creates standardized, CROMA-compatible RTC rasters:

    <instance-root>/s1_rtc_ready/<city>/<city>_s1_rtc_vv_vh_10m_aligned.tif

Output convention:

    band 1 = VV
    band 2 = VH

Important:
    - No VV_minus_VH is written.
    - Output is aligned exactly to the city S2 grid.
    - Output is float32.
    - Output has 2 bands only.
    - Output is intended for CROMA input shape [2, 224, 224].

Supported finalization routes:

    1. stacked_normal_reproject
        Source is a stacked georeferenced raster.
        Uses bands 1 and 2.
        Reprojects with normal CRS + affine transform.

    2. separate_vv_vh_gcp_reproject
        Source is separate VV and VH files.
        Uses band 1 from each file.
        Reprojects using GCPs.

Example dry run:

python src/rtc_processing/03_finalize_s1_rtc_ready.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --dry-run `
  --overwrite

Example real run:

python src/rtc_processing/03_finalize_s1_rtc_ready.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
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


def ensure_output_can_be_written(path: Path, overwrite: bool, skip_existing: bool) -> str:
    """
    Returns:
        "write" if output should be written
        "skip" if output exists and skip_existing=True
    """

    if path.exists():
        if skip_existing:
            return "skip"

        if not overwrite:
            fail(
                "Output already exists and neither --overwrite nor --skip-existing was provided:\n"
                f"  {path_to_str(path)}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    return "write"


def parse_bool(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


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


def normalize_city(value: str) -> str:
    return str(value).strip().replace("\\", "/").split("/")[-1]


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
    if path.exists() and not overwrite:
        fail(
            "Output CSV already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        fail(f"No rows to write for CSV: {path_to_str(path)}")

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output JSON already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(
    path: Path,
    summary: Dict[str, object],
    city_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output Markdown already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    lines.append("# S1 RTC ready finalization")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Georeferencing CSV: `{summary['georef_csv']}`")
    lines.append(f"- Output root: `{summary['output_root']}`")
    lines.append(f"- Dry run: `{summary['parameters']['dry_run']}`")
    lines.append(f"- Cities requested: `{summary['n_cities_requested']}`")
    lines.append(f"- Cities completed: `{summary['n_cities_completed']}`")
    lines.append(f"- Cities skipped existing: `{summary['n_cities_skipped_existing']}`")
    lines.append(f"- Cities failed: `{summary['n_cities_failed']}`")
    lines.append("")

    lines.append("## Route counts")
    lines.append("")
    lines.append("| route | cities |")
    lines.append("|---|---:|")
    for route, count in summary["route_counts"].items():
        lines.append(f"| {route} | {count} |")
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    outputs = summary["outputs"]
    lines.append(f"- City summary CSV: `{outputs['city_summary_csv']}`")
    lines.append(f"- JSON: `{outputs['json']}`")
    lines.append(f"- Markdown: `{outputs['markdown']}`")
    lines.append("")

    lines.append("## City-level status")
    lines.append("")
    lines.append(
        "| city | status | route | output | size | CRS | invalid before fill % | notes |"
    )
    lines.append("|---|---|---|---|---|---|---:|---|")

    for row in city_rows:
        size_text = ""
        if row["output_width"] != "" and row["output_height"] != "":
            size_text = f"{row['output_width']}×{row['output_height']}"

        lines.append(
            f"| {row['city']} | "
            f"{row['status']} | "
            f"{row['finalization_route']} | "
            f"`{row['output_path']}` | "
            f"{size_text} | "
            f"{row['output_crs']} | "
            f"{row['max_invalid_percent_before_fill']} | "
            f"{row['notes']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Every output raster is intended to have exactly 2 bands: VV and VH.")
    lines.append("- The output grid is copied from the corresponding S2 raster.")
    lines.append("- `stacked_normal_reproject` uses the source CRS and affine transform.")
    lines.append("- `separate_vv_vh_gcp_reproject` uses Ground Control Points from the VV and VH source rasters.")
    lines.append("- Invalid values before fill are NaN/Inf pixels created or propagated during reprojection.")
    lines.append("- Invalid values are filled with the configured `fill_invalid_value` before writing.")
    lines.append("- A separate validation script should be run next to verify alignment, band count, and valid-pixel statistics.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# S2 reference discovery
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
        fail(f"Missing S2 city folder for {city}: {path_to_str(city_dir)}")

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
            fail(
                f"Ambiguous S2 reference for city {city} using pattern {pattern}:\n"
                f"{formatted}"
            )

    fail(f"Could not find S2 reference raster for city {city} in {path_to_str(city_dir)}")


# ---------------------------------------------------------------------
# Reprojection helpers
# ---------------------------------------------------------------------

def resampling_from_name(name: str) -> Resampling:
    lookup = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }

    if name not in lookup:
        fail(f"Unsupported resampling: {name}")

    return lookup[name]


def get_gcps_and_crs(src) -> Tuple[list, object]:
    gcps, gcp_crs = src.gcps

    if not gcps or len(gcps) < 4 or gcp_crs is None:
        raise ValueError(
            f"Source does not have usable GCPs. gcp_count={len(gcps)}, gcp_crs={gcp_crs}"
        )

    return gcps, gcp_crs


def get_source_nodata(src) -> Optional[float]:
    nodata = src.nodata

    if nodata is None:
        return None

    try:
        return float(nodata)
    except Exception:
        return None


def reproject_normal_band(
    src,
    *,
    band_index: int,
    dst_shape: Tuple[int, int],
    dst_transform,
    dst_crs,
    resampling: Resampling,
    fill_initial: float,
    num_threads: int,
    warp_mem_limit: int,
) -> np.ndarray:
    dst = np.full(dst_shape, fill_initial, dtype=np.float32)

    src_nodata = get_source_nodata(src)

    kwargs = {
        "source": rasterio.band(src, band_index),
        "destination": dst,
        "src_transform": src.transform,
        "src_crs": src.crs,
        "src_nodata": src_nodata,
        "dst_transform": dst_transform,
        "dst_crs": dst_crs,
        "dst_nodata": fill_initial,
        "resampling": resampling,
        "num_threads": num_threads,
        "warp_mem_limit": warp_mem_limit,
    }

    if src_nodata is None:
        kwargs.pop("src_nodata")

    reproject(**kwargs)

    return dst


def reproject_gcp_band(
    src,
    *,
    band_index: int,
    dst_shape: Tuple[int, int],
    dst_transform,
    dst_crs,
    resampling: Resampling,
    fill_initial: float,
    num_threads: int,
    warp_mem_limit: int,
) -> np.ndarray:
    dst = np.full(dst_shape, fill_initial, dtype=np.float32)

    gcps, gcp_crs = get_gcps_and_crs(src)
    src_nodata = get_source_nodata(src)

    kwargs = {
        "source": rasterio.band(src, band_index),
        "destination": dst,
        "gcps": gcps,
        "src_crs": gcp_crs,
        "src_nodata": src_nodata,
        "dst_transform": dst_transform,
        "dst_crs": dst_crs,
        "dst_nodata": fill_initial,
        "resampling": resampling,
        "num_threads": num_threads,
        "warp_mem_limit": warp_mem_limit,
    }

    if src_nodata is None:
        kwargs.pop("src_nodata")

    reproject(**kwargs)

    return dst


def array_stats_before_fill(
    arr: np.ndarray,
    *,
    fill_initial: float,
) -> Dict[str, object]:
    total = int(arr.size)
    finite = np.isfinite(arr)

    invalid = ~finite

    # Treat the initial fill value as invalid if it remains in the destination.
    # This is diagnostic only. It helps detect uncovered pixels after reprojection.
    if np.isfinite(fill_initial):
        invalid |= arr == fill_initial

    invalid_pixels = int(np.count_nonzero(invalid))
    valid_pixels = int(total - invalid_pixels)
    invalid_percent = 100.0 * invalid_pixels / total if total else 0.0

    valid_values = arr[~invalid]

    if valid_values.size == 0:
        return {
            "total_pixels": total,
            "valid_pixels_before_fill": valid_pixels,
            "invalid_pixels_before_fill": invalid_pixels,
            "invalid_percent_before_fill": invalid_percent,
            "min_before_fill": "",
            "max_before_fill": "",
            "mean_before_fill": "",
        }

    return {
        "total_pixels": total,
        "valid_pixels_before_fill": valid_pixels,
        "invalid_pixels_before_fill": invalid_pixels,
        "invalid_percent_before_fill": invalid_percent,
        "min_before_fill": float(np.nanmin(valid_values)),
        "max_before_fill": float(np.nanmax(valid_values)),
        "mean_before_fill": float(np.nanmean(valid_values)),
    }


def fill_invalid_values(
    arr: np.ndarray,
    *,
    fill_initial: float,
    fill_invalid_value: float,
) -> np.ndarray:
    invalid = ~np.isfinite(arr)

    if np.isfinite(fill_initial):
        invalid |= arr == fill_initial

    out = arr.astype(np.float32, copy=True)
    out[invalid] = np.float32(fill_invalid_value)

    return out


# ---------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------

def build_output_profile(ref_src, *, compress: str, tiled: bool, block_size: int) -> Dict[str, object]:
    profile = ref_src.profile.copy()

    # Remove source-specific fields that should not control the RTC output.
    profile.pop("nodata", None)

    profile.update(
        {
            "driver": "GTiff",
            "count": 2,
            "dtype": "float32",
            "crs": ref_src.crs,
            "transform": ref_src.transform,
            "width": ref_src.width,
            "height": ref_src.height,
            "compress": compress,
            "BIGTIFF": "IF_SAFER",
        }
    )

    if tiled:
        profile.update(
            {
                "tiled": True,
                "blockxsize": block_size,
                "blockysize": block_size,
            }
        )
    else:
        profile.pop("tiled", None)
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)

    return profile


def write_two_band_output(
    output_path: Path,
    *,
    vv: np.ndarray,
    vh: np.ndarray,
    profile: Dict[str, object],
    overwrite: bool,
    skip_existing: bool,
) -> str:
    write_mode = ensure_output_can_be_written(output_path, overwrite, skip_existing)

    if write_mode == "skip":
        return "skipped_existing"

    tmp_path = output_path.with_name(
        output_path.stem + f".tmp_{os.getpid()}" + output_path.suffix
    )

    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(vv.astype(np.float32), 1)
            dst.write(vh.astype(np.float32), 2)
            dst.set_band_description(1, "VV")
            dst.set_band_description(2, "VH")

            dst.update_tags(
                product="S1_RTC_READY",
                bands="VV,VH",
                aligned_to="Instance C S2 grid",
                created_utc=datetime.now(timezone.utc).isoformat(),
            )

        if output_path.exists():
            output_path.unlink()

        tmp_path.replace(output_path)

        return "written"

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ---------------------------------------------------------------------
# Finalization per city
# ---------------------------------------------------------------------

def path_from_cell(value: object) -> Optional[Path]:
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    return Path(text)


def validate_georef_row(row: Dict[str, str]) -> None:
    city = row.get("city", "")
    usable = parse_bool(row.get("usable_for_finalization", ""))

    if not usable:
        fail(
            f"City {city} is marked as not usable in georeferencing CSV. "
            "Rerun 02_inspect_rtc_georeferencing.py and check outputs."
        )


def finalize_city(
    row: Dict[str, str],
    *,
    instance_root: Path,
    output_root: Path,
    resampling: Resampling,
    fill_initial: float,
    fill_invalid_value: float,
    fail_if_invalid_percent_gt: float,
    overwrite: bool,
    skip_existing: bool,
    dry_run: bool,
    compress: str,
    tiled: bool,
    block_size: int,
    num_threads: int,
    warp_mem_limit: int,
) -> Dict[str, object]:
    city = normalize_city(row["city"])
    route = str(row["finalization_route"]).strip()

    validate_georef_row(row)

    s2_path = find_s2_reference(instance_root, city)

    output_dir = output_root / city
    output_path = output_dir / f"{city}_s1_rtc_vv_vh_10m_aligned.tif"

    result: Dict[str, object] = {
        "city": city,
        "status": "not_started",
        "finalization_route": route,
        "s2_reference_path": path_to_str(s2_path),
        "source_stacked_path": row.get("stacked_path", ""),
        "source_vv_path": row.get("vv_path", ""),
        "source_vh_path": row.get("vh_path", ""),
        "output_path": path_to_str(output_path),
        "output_width": "",
        "output_height": "",
        "output_crs": "",
        "output_transform": "",
        "vv_invalid_pixels_before_fill": "",
        "vv_invalid_percent_before_fill": "",
        "vv_min_before_fill": "",
        "vv_max_before_fill": "",
        "vv_mean_before_fill": "",
        "vh_invalid_pixels_before_fill": "",
        "vh_invalid_percent_before_fill": "",
        "vh_min_before_fill": "",
        "vh_max_before_fill": "",
        "vh_mean_before_fill": "",
        "max_invalid_percent_before_fill": "",
        "fill_invalid_value": fill_invalid_value,
        "notes": "",
    }

    if dry_run:
        with rasterio.open(s2_path) as ref:
            result.update(
                {
                    "status": "dry_run",
                    "output_width": ref.width,
                    "output_height": ref.height,
                    "output_crs": str(ref.crs),
                    "output_transform": tuple(float(x) for x in (
                        ref.transform.a,
                        ref.transform.b,
                        ref.transform.c,
                        ref.transform.d,
                        ref.transform.e,
                        ref.transform.f,
                    )),
                    "notes": "Dry run only; no raster written.",
                }
            )
        return result

    write_mode = ensure_output_can_be_written(output_path, overwrite, skip_existing)

    if write_mode == "skip":
        with rasterio.open(s2_path) as ref:
            result.update(
                {
                    "status": "skipped_existing",
                    "output_width": ref.width,
                    "output_height": ref.height,
                    "output_crs": str(ref.crs),
                    "output_transform": tuple(float(x) for x in (
                        ref.transform.a,
                        ref.transform.b,
                        ref.transform.c,
                        ref.transform.d,
                        ref.transform.e,
                        ref.transform.f,
                    )),
                    "notes": "Output already exists and --skip-existing was used.",
                }
            )
        return result

    with rasterio.open(s2_path) as ref_src:
        dst_shape = (ref_src.height, ref_src.width)
        dst_transform = ref_src.transform
        dst_crs = ref_src.crs

        profile = build_output_profile(
            ref_src,
            compress=compress,
            tiled=tiled,
            block_size=block_size,
        )

        result.update(
            {
                "output_width": ref_src.width,
                "output_height": ref_src.height,
                "output_crs": str(ref_src.crs),
                "output_transform": tuple(float(x) for x in (
                    ref_src.transform.a,
                    ref_src.transform.b,
                    ref_src.transform.c,
                    ref_src.transform.d,
                    ref_src.transform.e,
                    ref_src.transform.f,
                )),
            }
        )

        if route == "stacked_normal_reproject":
            stacked_path = path_from_cell(row.get("stacked_path", ""))

            if stacked_path is None or not stacked_path.exists():
                raise FileNotFoundError(
                    f"{city}: stacked source path missing: {path_to_str(stacked_path)}"
                )

            with rasterio.open(stacked_path) as src:
                if src.count < 2:
                    raise ValueError(f"{city}: stacked source has fewer than 2 bands: {src.count}")

                if src.crs is None:
                    raise ValueError(f"{city}: stacked source has no CRS.")

                vv_raw = reproject_normal_band(
                    src,
                    band_index=1,
                    dst_shape=dst_shape,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    fill_initial=fill_initial,
                    num_threads=num_threads,
                    warp_mem_limit=warp_mem_limit,
                )

                vh_raw = reproject_normal_band(
                    src,
                    band_index=2,
                    dst_shape=dst_shape,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    fill_initial=fill_initial,
                    num_threads=num_threads,
                    warp_mem_limit=warp_mem_limit,
                )

        elif route == "separate_vv_vh_gcp_reproject":
            vv_path = path_from_cell(row.get("vv_path", ""))
            vh_path = path_from_cell(row.get("vh_path", ""))

            if vv_path is None or not vv_path.exists():
                raise FileNotFoundError(f"{city}: VV source path missing: {path_to_str(vv_path)}")

            if vh_path is None or not vh_path.exists():
                raise FileNotFoundError(f"{city}: VH source path missing: {path_to_str(vh_path)}")

            with rasterio.open(vv_path) as vv_src, rasterio.open(vh_path) as vh_src:
                vv_raw = reproject_gcp_band(
                    vv_src,
                    band_index=1,
                    dst_shape=dst_shape,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    fill_initial=fill_initial,
                    num_threads=num_threads,
                    warp_mem_limit=warp_mem_limit,
                )

                vh_raw = reproject_gcp_band(
                    vh_src,
                    band_index=1,
                    dst_shape=dst_shape,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    fill_initial=fill_initial,
                    num_threads=num_threads,
                    warp_mem_limit=warp_mem_limit,
                )

        else:
            raise ValueError(f"{city}: unsupported finalization route: {route}")

        vv_stats = array_stats_before_fill(vv_raw, fill_initial=fill_initial)
        vh_stats = array_stats_before_fill(vh_raw, fill_initial=fill_initial)

        max_invalid_percent = max(
            float(vv_stats["invalid_percent_before_fill"]),
            float(vh_stats["invalid_percent_before_fill"]),
        )

        result.update(
            {
                "vv_invalid_pixels_before_fill": vv_stats["invalid_pixels_before_fill"],
                "vv_invalid_percent_before_fill": round(float(vv_stats["invalid_percent_before_fill"]), 8),
                "vv_min_before_fill": vv_stats["min_before_fill"],
                "vv_max_before_fill": vv_stats["max_before_fill"],
                "vv_mean_before_fill": vv_stats["mean_before_fill"],
                "vh_invalid_pixels_before_fill": vh_stats["invalid_pixels_before_fill"],
                "vh_invalid_percent_before_fill": round(float(vh_stats["invalid_percent_before_fill"]), 8),
                "vh_min_before_fill": vh_stats["min_before_fill"],
                "vh_max_before_fill": vh_stats["max_before_fill"],
                "vh_mean_before_fill": vh_stats["mean_before_fill"],
                "max_invalid_percent_before_fill": round(float(max_invalid_percent), 8),
            }
        )

        if max_invalid_percent > fail_if_invalid_percent_gt:
            raise ValueError(
                f"{city}: invalid percent before fill is {max_invalid_percent:.6f}%, "
                f"which exceeds threshold {fail_if_invalid_percent_gt:.6f}%."
            )

        vv = fill_invalid_values(
            vv_raw,
            fill_initial=fill_initial,
            fill_invalid_value=fill_invalid_value,
        )

        vh = fill_invalid_values(
            vh_raw,
            fill_initial=fill_initial,
            fill_invalid_value=fill_invalid_value,
        )

        write_two_band_output(
            output_path,
            vv=vv,
            vh=vh,
            profile=profile,
            overwrite=overwrite,
            skip_existing=skip_existing,
        )

        result["status"] = "completed"
        result["notes"] = "Wrote 2-band VV/VH RTC raster aligned to S2 grid."

    return result


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    *,
    instance_root: Path,
    georef_csv: Path,
    output_root: Path,
    city_results: List[Dict[str, object]],
    args: argparse.Namespace,
    output_csv: Path,
    output_json: Path,
    output_md: Path,
) -> Dict[str, object]:
    route_counts: Dict[str, int] = {}

    for row in city_results:
        route = str(row["finalization_route"])
        route_counts[route] = route_counts.get(route, 0) + 1

    n_completed = sum(1 for row in city_results if row["status"] == "completed")
    n_skipped = sum(1 for row in city_results if row["status"] == "skipped_existing")
    n_failed = sum(1 for row in city_results if row["status"] == "failed")
    n_dry_run = sum(1 for row in city_results if row["status"] == "dry_run")

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "georef_csv": path_to_str(georef_csv),
        "output_root": path_to_str(output_root),
        "n_cities_requested": len(city_results),
        "n_cities_completed": n_completed,
        "n_cities_skipped_existing": n_skipped,
        "n_cities_failed": n_failed,
        "n_cities_dry_run": n_dry_run,
        "route_counts": dict(sorted(route_counts.items())),
        "parameters": {
            "resampling": args.resampling,
            "fill_initial": args.fill_initial,
            "fill_invalid_value": args.fill_invalid_value,
            "fail_if_invalid_percent_gt": args.fail_if_invalid_percent_gt,
            "compress": args.compress,
            "tiled": bool(args.tiled),
            "block_size": args.block_size,
            "num_threads": args.num_threads,
            "warp_mem_limit": args.warp_mem_limit,
            "dry_run": bool(args.dry_run),
            "overwrite": bool(args.overwrite),
            "skip_existing": bool(args.skip_existing),
        },
        "outputs": {
            "city_summary_csv": path_to_str(output_csv),
            "json": path_to_str(output_json),
            "markdown": path_to_str(output_md),
        },
        "city_results": city_results,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize S1 RTC rasters into Instance C as 2-band VV/VH aligned outputs."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--georef-csv",
        type=Path,
        default=None,
        help=(
            "Optional georeferencing CSV. "
            "Default: <instance-root>/metadata/rtc_processing/rtc_georeferencing_by_city.csv"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output root. "
            "Default: <instance-root>/s1_rtc_ready"
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help=(
            "Optional report output directory. "
            "Default: <instance-root>/metadata/rtc_processing"
        ),
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Optional subset of cities to process.",
    )

    parser.add_argument(
        "--expected-city-count",
        type=int,
        default=26,
        help="Expected number of cities when --cities is not used. Default: 26.",
    )

    parser.add_argument(
        "--no-require-expected-city-count",
        action="store_true",
        help="Warn instead of failing if full run city count is not expected-city-count.",
    )

    parser.add_argument(
        "--resampling",
        choices=["nearest", "bilinear", "cubic", "average"],
        default="bilinear",
        help="Resampling method for RTC reprojection. Default: bilinear.",
    )

    parser.add_argument(
        "--fill-initial",
        type=float,
        default=-9999.0,
        help=(
            "Internal destination fill value before reprojection. "
            "Pixels still equal to this after reprojection are considered invalid. Default: -9999."
        ),
    )

    parser.add_argument(
        "--fill-invalid-value",
        type=float,
        default=0.0,
        help="Value used to fill invalid pixels before writing output. Default: 0.0.",
    )

    parser.add_argument(
        "--fail-if-invalid-percent-gt",
        type=float,
        default=100.0,
        help=(
            "Fail a city if max invalid percent before fill is greater than this value. "
            "Default: 100.0, meaning do not fail by invalid percentage."
        ),
    )

    parser.add_argument(
        "--compress",
        choices=["deflate", "lzw", "none"],
        default="deflate",
        help="GeoTIFF compression. Default: deflate.",
    )

    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Write tiled GeoTIFF. Recommended.",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Tile block size when --tiled is used. Default: 512.",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="GDAL warp threads per reprojection. Default: 2.",
    )

    parser.add_argument(
        "--warp-mem-limit",
        type=int,
        default=512,
        help="GDAL warp memory limit in MB. Default: 512.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check inputs and planned outputs without writing rasters.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cities whose output already exists.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output rasters and reports.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root

    report_dir: Path = args.report_dir or (
        instance_root / "metadata" / "rtc_processing"
    )

    georef_csv: Path = args.georef_csv or (
        report_dir / "rtc_georeferencing_by_city.csv"
    )

    output_root: Path = args.output_root or (
        instance_root / "s1_rtc_ready"
    )

    output_csv = report_dir / "s1_rtc_ready_finalization.csv"
    output_json = report_dir / "s1_rtc_ready_finalization.json"
    output_md = report_dir / "s1_rtc_ready_finalization.md"

    log("STEP", "Finalizing S1 RTC ready rasters.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Georef CSV:    {path_to_str(georef_csv)}")
    log("INFO", f"Output root:   {path_to_str(output_root)}")
    log("INFO", f"Report dir:    {path_to_str(report_dir)}")
    log("INFO", f"Dry run:       {args.dry_run}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    georef_rows = read_csv_rows(georef_csv)

    if args.cities:
        requested = set(normalize_city(city) for city in args.cities)
        before = len(georef_rows)
        georef_rows = [
            row for row in georef_rows
            if normalize_city(row["city"]) in requested
        ]
        after = len(georef_rows)

        found = set(normalize_city(row["city"]) for row in georef_rows)
        missing = sorted(requested - found)

        if missing:
            fail(
                "Requested cities were not found in georeferencing CSV:\n"
                + "\n".join(f"  - {city}" for city in missing)
            )

        log("WARN", f"City subset enabled. Rows before={before}, after={after}")

    if not args.cities and len(georef_rows) != args.expected_city_count:
        msg = (
            f"Georeferencing CSV contains {len(georef_rows)} cities, "
            f"expected {args.expected_city_count}."
        )

        if args.no_require_expected_city_count:
            log("WARN", msg)
        else:
            fail(msg + " Use --no-require-expected-city-count if intentional.")

    resampling = resampling_from_name(args.resampling)

    city_results: List[Dict[str, object]] = []

    for row in georef_rows:
        city = normalize_city(row["city"])
        log("STEP", f"Finalizing city: {city}")

        try:
            result = finalize_city(
                row,
                instance_root=instance_root,
                output_root=output_root,
                resampling=resampling,
                fill_initial=float(args.fill_initial),
                fill_invalid_value=float(args.fill_invalid_value),
                fail_if_invalid_percent_gt=float(args.fail_if_invalid_percent_gt),
                overwrite=bool(args.overwrite),
                skip_existing=bool(args.skip_existing),
                dry_run=bool(args.dry_run),
                compress=("DEFLATE" if args.compress == "deflate" else "LZW" if args.compress == "lzw" else ""),
                tiled=bool(args.tiled),
                block_size=int(args.block_size),
                num_threads=int(args.num_threads),
                warp_mem_limit=int(args.warp_mem_limit),
            )

            city_results.append(result)

            log(
                "OK",
                f"{city}: status={result['status']}, "
                f"route={result['finalization_route']}, "
                f"max_invalid_before_fill={result['max_invalid_percent_before_fill']}",
            )

        except Exception as exc:
            result = {
                "city": city,
                "status": "failed",
                "finalization_route": row.get("finalization_route", ""),
                "s2_reference_path": "",
                "source_stacked_path": row.get("stacked_path", ""),
                "source_vv_path": row.get("vv_path", ""),
                "source_vh_path": row.get("vh_path", ""),
                "output_path": path_to_str(output_root / city / f"{city}_s1_rtc_vv_vh_10m_aligned.tif"),
                "output_width": "",
                "output_height": "",
                "output_crs": "",
                "output_transform": "",
                "vv_invalid_pixels_before_fill": "",
                "vv_invalid_percent_before_fill": "",
                "vv_min_before_fill": "",
                "vv_max_before_fill": "",
                "vv_mean_before_fill": "",
                "vh_invalid_pixels_before_fill": "",
                "vh_invalid_percent_before_fill": "",
                "vh_min_before_fill": "",
                "vh_max_before_fill": "",
                "vh_mean_before_fill": "",
                "max_invalid_percent_before_fill": "",
                "fill_invalid_value": args.fill_invalid_value,
                "notes": repr(exc),
            }

            city_results.append(result)
            log("ERROR", f"{city}: failed with error: {repr(exc)}")

    summary = build_summary(
        instance_root=instance_root,
        georef_csv=georef_csv,
        output_root=output_root,
        city_results=city_results,
        args=args,
        output_csv=output_csv,
        output_json=output_json,
        output_md=output_md,
    )

    log("STEP", "Writing finalization reports.")

    write_csv(output_csv, city_results, overwrite=True)
    write_json(output_json, summary, overwrite=True)
    write_markdown(output_md, summary, city_results, overwrite=True)

    log("OK", f"Wrote CSV:      {path_to_str(output_csv)}")
    log("OK", f"Wrote JSON:     {path_to_str(output_json)}")
    log("OK", f"Wrote Markdown: {path_to_str(output_md)}")

    log("STEP", "Final summary.")
    log("OK", f"Cities requested: {summary['n_cities_requested']}")
    log("OK", f"Cities completed: {summary['n_cities_completed']}")
    log("OK", f"Cities skipped existing: {summary['n_cities_skipped_existing']}")
    log("OK", f"Cities failed: {summary['n_cities_failed']}")
    log("INFO", "Route counts:")
    for route, count in summary["route_counts"].items():
        log("INFO", f"  {route}: {count}")

    if summary["n_cities_failed"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()