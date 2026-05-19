#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_merge_rtc_fallback.py

Merge useful RTC fallback candidates into finalized Instance C RTC rasters.

This script is designed after running:

    04_validate_s1_rtc_ready.py
    05_find_rtc_fallback_candidates.py

It repairs finalized RTC rasters by replacing current all-zero VV/VH pixels
with valid values from a fallback RTC candidate.

Important:
    - This script only fills pixels where the current RTC has both VV and VH equal to zero.
    - It does NOT overwrite valid current RTC pixels.
    - It uses the current RTC raster as the reference grid.
    - It writes 2-band VV/VH output.
    - It can safely backup the current RTC before replacing it.

Recommended first use:
    Duque de Caxias only, because fallback discovery showed:
        fillable_current_zero_percent = 100%
        fillable_label_zero_percent   = 100%

Example dry run:

python src/rtc_processing/06_merge_rtc_fallback.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --cities duque_de_caxias `
  --dry-run `
  --overwrite

Example real merge, replacing current s1_rtc_ready safely with backup:

python src/rtc_processing/06_merge_rtc_fallback.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --cities duque_de_caxias `
  --replace-current `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def normalize_city(value: str) -> str:
    value = str(value).strip()
    value = value.replace("\\", "/").split("/")[-1]
    value = value.lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


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


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


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
    rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# RTC fallback merge report")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Candidate CSV: `{summary['candidate_csv']}`")
    lines.append(f"- Output root: `{summary['output_root']}`")
    lines.append(f"- Replace current: `{summary['parameters']['replace_current']}`")
    lines.append(f"- Dry run: `{summary['parameters']['dry_run']}`")
    lines.append(f"- Cities requested: `{summary['n_cities_requested']}`")
    lines.append(f"- Cities completed: `{summary['n_cities_completed']}`")
    lines.append(f"- Cities failed: `{summary['n_cities_failed']}`")
    lines.append("")

    lines.append("## City-level merge results")
    lines.append("")
    lines.append(
        "| city | status | candidate | current zero before % | candidate-fillable zero % | "
        "zero after % | repaired pixels | repaired label-zero pixels | output | notes |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---|")

    for row in rows:
        lines.append(
            f"| {row['city']} | "
            f"{row['status']} | "
            f"`{row['candidate_id']}` | "
            f"{row['current_zero_percent_before']} | "
            f"{row['candidate_fillable_current_zero_percent_reported']} | "
            f"{row['current_zero_percent_after']} | "
            f"{row['repaired_zero_pixels']} | "
            f"{row['repaired_label_zero_pixels']} | "
            f"`{row['output_path']}` | "
            f"{row['notes']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The merge only replaces pixels where the current RTC has both VV and VH equal to zero.")
    lines.append("- Valid non-zero RTC pixels in the current raster are preserved.")
    lines.append("- The fallback candidate is reprojected to the exact current RTC/S2 grid before merging.")
    lines.append("- If `replace_current=True`, the original RTC file is copied to the backup folder before replacement.")
    lines.append("- After this script, rerun `04_validate_s1_rtc_ready.py` to verify the repair.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Raster path helpers
# ---------------------------------------------------------------------

def current_rtc_path(instance_root: Path, city: str) -> Path:
    city = normalize_city(city)
    return instance_root / "s1_rtc_ready" / city / f"{city}_s1_rtc_vv_vh_10m_aligned.tif"


def label_path(instance_root: Path, city: str) -> Path:
    city = normalize_city(city)
    label_dir = instance_root / "labels" / city

    if not label_dir.exists():
        fail(f"Missing label folder for {city}: {path_to_str(label_dir)}")

    candidates = sorted(label_dir.glob(f"{city}_label_final.tif"))

    if len(candidates) == 1:
        return candidates[0]

    candidates = sorted(label_dir.glob("*label*.tif"))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        fail(
            f"Ambiguous label raster for {city}:\n"
            + "\n".join(f"  - {path_to_str(p)}" for p in candidates)
        )

    fail(f"Could not find label raster for {city} in {path_to_str(label_dir)}")


# ---------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------

def load_best_candidates(
    candidate_csv: Path,
    *,
    cities: Optional[Sequence[str]],
    min_fillable_current_zero_percent: float,
    min_fillable_label_zero_percent: float,
) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(candidate_csv)

    if cities:
        requested = set(normalize_city(c) for c in cities)
        rows = [
            row for row in rows
            if normalize_city(row["city"]) in requested
        ]

        found = set(normalize_city(row["city"]) for row in rows)
        missing = sorted(requested - found)

        if missing:
            fail(
                "Requested cities were not found in candidate CSV:\n"
                + "\n".join(f"  - {city}" for city in missing)
            )

    usable = [
        row for row in rows
        if row.get("status", "") == "ok"
        and (
            safe_float(row.get("fillable_current_zero_percent", 0.0)) >= min_fillable_current_zero_percent
            or safe_float(row.get("fillable_label_zero_percent", 0.0)) >= min_fillable_label_zero_percent
        )
    ]

    if not usable:
        fail(
            "No usable fallback candidates passed thresholds.\n"
            f"Candidate CSV: {path_to_str(candidate_csv)}"
        )

    by_city: Dict[str, List[Dict[str, str]]] = {}

    for row in usable:
        city = normalize_city(row["city"])
        by_city.setdefault(city, []).append(row)

    best: Dict[str, Dict[str, str]] = {}

    for city, city_rows in by_city.items():
        city_rows = sorted(
            city_rows,
            key=lambda row: (
                -safe_float(row.get("fillable_label_zero_percent", 0.0)),
                -safe_float(row.get("fillable_current_zero_percent", 0.0)),
                safe_int(row.get("is_primary_source", 0), 0),
                row.get("candidate_id", ""),
            ),
        )

        best[city] = city_rows[0]

    return best


# ---------------------------------------------------------------------
# Georeferencing helpers
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


def is_identity_like_transform(transform, tolerance: float = 1e-9) -> bool:
    a, b, c, d, e, f = affine_six(transform)

    return (
        abs(a - 1.0) <= tolerance
        and abs(b) <= tolerance
        and abs(c) <= tolerance
        and abs(d) <= tolerance
        and abs(e - 1.0) <= tolerance
        and abs(f) <= tolerance
    )


def has_normal_georef(src) -> bool:
    if src.crs is None:
        return False

    if is_identity_like_transform(src.transform):
        return False

    return True


def has_usable_gcps(src) -> bool:
    gcps, gcp_crs = src.gcps
    return bool(gcps and len(gcps) >= 4 and gcp_crs is not None)


def get_source_nodata(src) -> Optional[float]:
    if src.nodata is None:
        return None

    try:
        return float(src.nodata)
    except Exception:
        return None


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


def reproject_band_to_reference(
    src,
    *,
    band_index: int,
    ref,
    fill_initial: float,
    resampling: Resampling,
    num_threads: int,
    warp_mem_limit: int,
) -> np.ndarray:
    dst = np.full((ref.height, ref.width), fill_initial, dtype=np.float32)

    kwargs = {
        "source": rasterio.band(src, band_index),
        "destination": dst,
        "dst_transform": ref.transform,
        "dst_crs": ref.crs,
        "dst_nodata": fill_initial,
        "resampling": resampling,
        "num_threads": num_threads,
        "warp_mem_limit": warp_mem_limit,
    }

    src_nodata = get_source_nodata(src)

    if src_nodata is not None:
        kwargs["src_nodata"] = src_nodata

    if has_normal_georef(src):
        kwargs["src_transform"] = src.transform
        kwargs["src_crs"] = src.crs
    elif has_usable_gcps(src):
        gcps, gcp_crs = src.gcps
        kwargs["gcps"] = gcps
        kwargs["src_crs"] = gcp_crs
    else:
        raise ValueError("Candidate source has neither normal georeferencing nor usable GCPs.")

    reproject(**kwargs)

    return dst


# ---------------------------------------------------------------------
# Mask/stat helpers
# ---------------------------------------------------------------------

def read_current_rtc(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    if not path.exists():
        fail(f"Current RTC raster does not exist: {path_to_str(path)}")

    with rasterio.open(path) as src:
        if src.count != 2:
            fail(f"Current RTC raster should have 2 bands, got {src.count}: {path_to_str(path)}")

        arr = src.read([1, 2]).astype(np.float32)
        profile = src.profile.copy()
        tags = src.tags()
        descriptions = src.descriptions

    meta = {
        "profile": profile,
        "tags": tags,
        "descriptions": descriptions,
    }

    return arr, meta


def read_label_positive(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        label = src.read(1, masked=True)

    data = np.ma.getdata(label)
    mask = np.ma.getmaskarray(label)

    if mask.shape == ():
        mask = np.zeros(data.shape, dtype=bool)

    valid = ~mask

    if np.issubdtype(data.dtype, np.floating):
        valid &= np.isfinite(data)

    return valid & (data > 0)


def current_zero_mask(arr: np.ndarray, zero_epsilon: float) -> np.ndarray:
    return (
        np.isfinite(arr[0])
        & np.isfinite(arr[1])
        & (np.abs(arr[0]) <= zero_epsilon)
        & (np.abs(arr[1]) <= zero_epsilon)
    )


def candidate_valid_mask(
    vv: np.ndarray,
    vh: np.ndarray,
    *,
    fill_initial: float,
    zero_epsilon: float,
    candidate_zero_as_invalid: bool,
) -> np.ndarray:
    valid = np.isfinite(vv) & np.isfinite(vh)

    if np.isfinite(fill_initial):
        valid &= vv != fill_initial
        valid &= vh != fill_initial

    if candidate_zero_as_invalid:
        both_zero = (np.abs(vv) <= zero_epsilon) & (np.abs(vh) <= zero_epsilon)
        valid &= ~both_zero

    return valid


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


# ---------------------------------------------------------------------
# Candidate reprojection
# ---------------------------------------------------------------------

def reproject_candidate_to_current_grid(
    candidate: Dict[str, str],
    *,
    ref_path: Path,
    fill_initial: float,
    resampling: Resampling,
    num_threads: int,
    warp_mem_limit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    candidate_type = candidate["candidate_type"]

    with rasterio.open(ref_path) as ref:
        if candidate_type == "stacked":
            stacked_path = Path(candidate["stacked_path"])

            if not stacked_path.exists():
                raise FileNotFoundError(f"Stacked candidate not found: {path_to_str(stacked_path)}")

            with rasterio.open(stacked_path) as src:
                if src.count < 2:
                    raise ValueError(f"Stacked candidate has fewer than 2 bands: {src.count}")

                vv = reproject_band_to_reference(
                    src,
                    band_index=1,
                    ref=ref,
                    fill_initial=fill_initial,
                    resampling=resampling,
                    num_threads=num_threads,
                    warp_mem_limit=warp_mem_limit,
                )

                vh = reproject_band_to_reference(
                    src,
                    band_index=2,
                    ref=ref,
                    fill_initial=fill_initial,
                    resampling=resampling,
                    num_threads=num_threads,
                    warp_mem_limit=warp_mem_limit,
                )

        elif candidate_type == "separate_vv_vh":
            vv_path = Path(candidate["vv_path"])
            vh_path = Path(candidate["vh_path"])

            if not vv_path.exists():
                raise FileNotFoundError(f"VV candidate not found: {path_to_str(vv_path)}")

            if not vh_path.exists():
                raise FileNotFoundError(f"VH candidate not found: {path_to_str(vh_path)}")

            with rasterio.open(vv_path) as vv_src:
                vv = reproject_band_to_reference(
                    vv_src,
                    band_index=1,
                    ref=ref,
                    fill_initial=fill_initial,
                    resampling=resampling,
                    num_threads=num_threads,
                    warp_mem_limit=warp_mem_limit,
                )

            with rasterio.open(vh_path) as vh_src:
                vh = reproject_band_to_reference(
                    vh_src,
                    band_index=1,
                    ref=ref,
                    fill_initial=fill_initial,
                    resampling=resampling,
                    num_threads=num_threads,
                    warp_mem_limit=warp_mem_limit,
                )

        else:
            raise ValueError(f"Unsupported candidate type: {candidate_type}")

    return vv, vh


# ---------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------

def write_rtc_output(
    path: Path,
    *,
    arr: np.ndarray,
    meta: Dict[str, object],
    merge_tags: Dict[str, str],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    tmp_path = path.with_name(path.stem + f".tmp_{os.getpid()}" + path.suffix)

    if tmp_path.exists():
        tmp_path.unlink()

    profile = meta["profile"].copy()
    profile.update(
        {
            "driver": "GTiff",
            "count": 2,
            "dtype": "float32",
            "BIGTIFF": "IF_SAFER",
        }
    )

    profile.pop("nodata", None)

    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(arr[0].astype(np.float32), 1)
            dst.write(arr[1].astype(np.float32), 2)

            descriptions = meta.get("descriptions", ("VV", "VH"))
            desc1 = descriptions[0] if len(descriptions) > 0 and descriptions[0] else "VV"
            desc2 = descriptions[1] if len(descriptions) > 1 and descriptions[1] else "VH"

            dst.set_band_description(1, desc1)
            dst.set_band_description(2, desc2)

            old_tags = meta.get("tags", {})
            dst.update_tags(**old_tags)
            dst.update_tags(**merge_tags)

        if path.exists():
            path.unlink()

        tmp_path.replace(path)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_merge_mask(
    path: Path,
    *,
    merge_mask: np.ndarray,
    ref_path: Path,
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "count": 1,
                "dtype": "uint8",
                "compress": "DEFLATE",
                "BIGTIFF": "IF_SAFER",
            }
        )
        profile.pop("nodata", None)

    tmp_path = path.with_name(path.stem + f".tmp_{os.getpid()}" + path.suffix)

    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(merge_mask.astype(np.uint8), 1)
            dst.set_band_description(1, "fallback_merge_mask")
            dst.update_tags(
                meaning="1 where fallback candidate replaced current all-zero RTC pixels",
                created_utc=datetime.now(timezone.utc).isoformat(),
            )

        if path.exists():
            path.unlink()

        tmp_path.replace(path)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def backup_current_file(
    current_path: Path,
    backup_root: Path,
    city: str,
    overwrite: bool,
) -> Path:
    city = normalize_city(city)
    backup_path = backup_root / city / current_path.name

    if backup_path.exists() and not overwrite:
        fail(
            "Backup already exists and --overwrite was not provided:\n"
            f"  {path_to_str(backup_path)}"
        )

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current_path, backup_path)

    return backup_path


# ---------------------------------------------------------------------
# Merge per city
# ---------------------------------------------------------------------

def merge_city(
    city: str,
    candidate: Dict[str, str],
    *,
    instance_root: Path,
    output_root: Path,
    backup_root: Path,
    mask_root: Path,
    replace_current: bool,
    dry_run: bool,
    overwrite: bool,
    fill_initial: float,
    zero_epsilon: float,
    candidate_zero_as_invalid: bool,
    resampling: Resampling,
    num_threads: int,
    warp_mem_limit: int,
) -> Dict[str, object]:
    city = normalize_city(city)

    curr_path = current_rtc_path(instance_root, city)
    lab_path = label_path(instance_root, city)

    current_arr, current_meta = read_current_rtc(curr_path)
    label_positive = read_label_positive(lab_path)

    zero_before = current_zero_mask(current_arr, zero_epsilon)
    zero_before_pixels = int(np.count_nonzero(zero_before))
    total_pixels = int(zero_before.size)
    zero_before_percent = percent(zero_before_pixels, total_pixels)

    label_zero_before = zero_before & label_positive
    label_zero_before_pixels = int(np.count_nonzero(label_zero_before))

    vv_fallback, vh_fallback = reproject_candidate_to_current_grid(
        candidate,
        ref_path=curr_path,
        fill_initial=fill_initial,
        resampling=resampling,
        num_threads=num_threads,
        warp_mem_limit=warp_mem_limit,
    )

    valid_fallback = candidate_valid_mask(
        vv_fallback,
        vh_fallback,
        fill_initial=fill_initial,
        zero_epsilon=zero_epsilon,
        candidate_zero_as_invalid=candidate_zero_as_invalid,
    )

    merge_mask = zero_before & valid_fallback

    repaired_pixels = int(np.count_nonzero(merge_mask))
    repaired_label_zero_pixels = int(np.count_nonzero(merge_mask & label_positive))

    merged = current_arr.copy()
    merged[0][merge_mask] = vv_fallback[merge_mask].astype(np.float32)
    merged[1][merge_mask] = vh_fallback[merge_mask].astype(np.float32)

    zero_after = current_zero_mask(merged, zero_epsilon)
    zero_after_pixels = int(np.count_nonzero(zero_after))
    zero_after_percent = percent(zero_after_pixels, total_pixels)

    label_zero_after_pixels = int(np.count_nonzero(zero_after & label_positive))

    if replace_current:
        out_path = curr_path
    else:
        out_path = output_root / city / curr_path.name

    mask_path = mask_root / city / f"{city}_fallback_merge_mask.tif"

    backup_path = ""

    status = "dry_run" if dry_run else "completed"

    if not dry_run:
        if replace_current:
            backup = backup_current_file(
                current_path=curr_path,
                backup_root=backup_root,
                city=city,
                overwrite=overwrite,
            )
            backup_path = path_to_str(backup)

        merge_tags = {
            "fallback_merge": "true",
            "fallback_candidate_id": str(candidate["candidate_id"]),
            "fallback_candidate_type": str(candidate["candidate_type"]),
            "fallback_merge_created_utc": datetime.now(timezone.utc).isoformat(),
            "fallback_repaired_pixels": str(repaired_pixels),
            "fallback_repaired_label_zero_pixels": str(repaired_label_zero_pixels),
        }

        write_rtc_output(
            out_path,
            arr=merged,
            meta=current_meta,
            merge_tags=merge_tags,
            overwrite=overwrite,
        )

        write_merge_mask(
            mask_path,
            merge_mask=merge_mask,
            ref_path=curr_path if replace_current else out_path,
            overwrite=overwrite,
        )

    return {
        "city": city,
        "status": status,
        "candidate_id": candidate["candidate_id"],
        "candidate_type": candidate["candidate_type"],
        "candidate_reported_status": candidate.get("status", ""),
        "candidate_fillable_current_zero_percent_reported": candidate.get("fillable_current_zero_percent", ""),
        "candidate_fillable_label_zero_percent_reported": candidate.get("fillable_label_zero_percent", ""),
        "stacked_path": candidate.get("stacked_path", ""),
        "vv_path": candidate.get("vv_path", ""),
        "vh_path": candidate.get("vh_path", ""),
        "current_rtc_path": path_to_str(curr_path),
        "label_path": path_to_str(lab_path),
        "output_path": path_to_str(out_path),
        "backup_path": backup_path,
        "merge_mask_path": path_to_str(mask_path),
        "total_pixels": total_pixels,
        "current_zero_pixels_before": zero_before_pixels,
        "current_zero_percent_before": round(zero_before_percent, 8),
        "label_zero_pixels_before": label_zero_before_pixels,
        "repaired_zero_pixels": repaired_pixels,
        "repaired_zero_percent_of_current_zero": round(percent(repaired_pixels, zero_before_pixels), 8),
        "repaired_label_zero_pixels": repaired_label_zero_pixels,
        "repaired_label_zero_percent_of_label_zero": round(percent(repaired_label_zero_pixels, label_zero_before_pixels), 8),
        "current_zero_pixels_after": zero_after_pixels,
        "current_zero_percent_after": round(zero_after_percent, 8),
        "label_zero_pixels_after": label_zero_after_pixels,
        "notes": (
            "Dry run only; no raster written."
            if dry_run
            else "Merged fallback into current RTC all-zero pixels."
        ),
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    *,
    instance_root: Path,
    candidate_csv: Path,
    output_root: Path,
    rows: List[Dict[str, object]],
    args: argparse.Namespace,
    csv_path: Path,
    json_path: Path,
    md_path: Path,
) -> Dict[str, object]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "candidate_csv": path_to_str(candidate_csv),
        "output_root": path_to_str(output_root),
        "n_cities_requested": len(rows),
        "n_cities_completed": sum(1 for r in rows if r["status"] == "completed"),
        "n_cities_dry_run": sum(1 for r in rows if r["status"] == "dry_run"),
        "n_cities_failed": sum(1 for r in rows if r["status"] == "failed"),
        "parameters": {
            "replace_current": bool(args.replace_current),
            "dry_run": bool(args.dry_run),
            "min_fillable_current_zero_percent": args.min_fillable_current_zero_percent,
            "min_fillable_label_zero_percent": args.min_fillable_label_zero_percent,
            "fill_initial": args.fill_initial,
            "zero_epsilon": args.zero_epsilon,
            "candidate_zero_as_invalid": bool(args.candidate_zero_as_invalid),
            "resampling": args.resampling,
            "num_threads": args.num_threads,
            "warp_mem_limit": args.warp_mem_limit,
        },
        "outputs": {
            "csv": path_to_str(csv_path),
            "json": path_to_str(json_path),
            "markdown": path_to_str(md_path),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge RTC fallback candidates into zero-filled current RTC-ready rasters."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=None,
        help=(
            "Optional fallback candidate CSV. "
            "Default: <instance-root>/metadata/rtc_processing/fallback_candidates/"
            "rtc_fallback_candidate_summary.csv"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output root used when --replace-current is not set. "
            "Default: <instance-root>/s1_rtc_ready_fallback_merged"
        ),
    )

    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help=(
            "Backup root used when --replace-current is set. "
            "Default: <instance-root>/metadata/rtc_processing/backups/s1_rtc_ready_before_fallback_merge"
        ),
    )

    parser.add_argument(
        "--mask-root",
        type=Path,
        default=None,
        help=(
            "Output root for fallback merge masks. "
            "Default: <instance-root>/metadata/rtc_processing/fallback_merge_masks"
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help=(
            "Report directory. "
            "Default: <instance-root>/metadata/rtc_processing/fallback_merge"
        ),
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Cities to merge. Recommended first use: --cities duque_de_caxias.",
    )

    parser.add_argument(
        "--min-fillable-current-zero-percent",
        type=float,
        default=10.0,
        help="Minimum fillable current zero percent. Default: 10.",
    )

    parser.add_argument(
        "--min-fillable-label-zero-percent",
        type=float,
        default=1.0,
        help="Minimum fillable label-zero percent. Default: 1.",
    )

    parser.add_argument(
        "--replace-current",
        action="store_true",
        help="Replace files inside s1_rtc_ready after creating a backup.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full merge computation but do not write rasters.",
    )

    parser.add_argument(
        "--fill-initial",
        type=float,
        default=-9999.0,
        help="Internal fill value for fallback reprojection. Default: -9999.",
    )

    parser.add_argument(
        "--zero-epsilon",
        type=float,
        default=1e-6,
        help="Tolerance for treating VV/VH as zero. Default: 1e-6.",
    )

    parser.add_argument(
        "--candidate-zero-as-invalid",
        action="store_true",
        default=True,
        help="Treat fallback pixels with both VV/VH equal zero as invalid. Default: enabled.",
    )

    parser.add_argument(
        "--resampling",
        choices=["nearest", "bilinear", "cubic", "average"],
        default="bilinear",
        help="Fallback reprojection resampling. Default: bilinear.",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="GDAL warp threads. Default: 2.",
    )

    parser.add_argument(
        "--warp-mem-limit",
        type=int,
        default=512,
        help="GDAL warp memory limit in MB. Default: 512.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs/reports/backups if they already exist.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root

    candidate_csv: Path = args.candidate_csv or (
        instance_root
        / "metadata"
        / "rtc_processing"
        / "fallback_candidates"
        / "rtc_fallback_candidate_summary.csv"
    )

    output_root: Path = args.output_root or (
        instance_root / "s1_rtc_ready_fallback_merged"
    )

    backup_root: Path = args.backup_root or (
        instance_root
        / "metadata"
        / "rtc_processing"
        / "backups"
        / "s1_rtc_ready_before_fallback_merge"
    )

    mask_root: Path = args.mask_root or (
        instance_root
        / "metadata"
        / "rtc_processing"
        / "fallback_merge_masks"
    )

    report_dir: Path = args.report_dir or (
        instance_root
        / "metadata"
        / "rtc_processing"
        / "fallback_merge"
    )

    csv_path = report_dir / "rtc_fallback_merge_summary.csv"
    json_path = report_dir / "rtc_fallback_merge_report.json"
    md_path = report_dir / "rtc_fallback_merge_report.md"

    log("STEP", "Merging RTC fallback candidates.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Candidate CSV: {path_to_str(candidate_csv)}")
    log("INFO", f"Output root:   {path_to_str(output_root)}")
    log("INFO", f"Backup root:   {path_to_str(backup_root)}")
    log("INFO", f"Mask root:     {path_to_str(mask_root)}")
    log("INFO", f"Replace current: {args.replace_current}")
    log("INFO", f"Dry run: {args.dry_run}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    candidates = load_best_candidates(
        candidate_csv,
        cities=args.cities,
        min_fillable_current_zero_percent=float(args.min_fillable_current_zero_percent),
        min_fillable_label_zero_percent=float(args.min_fillable_label_zero_percent),
    )

    resampling = resampling_from_name(args.resampling)

    rows: List[Dict[str, object]] = []

    for city, candidate in sorted(candidates.items()):
        log("STEP", f"Merging city: {city}")
        log("INFO", f"Candidate: {candidate['candidate_id']}")

        try:
            row = merge_city(
                city,
                candidate,
                instance_root=instance_root,
                output_root=output_root,
                backup_root=backup_root,
                mask_root=mask_root,
                replace_current=bool(args.replace_current),
                dry_run=bool(args.dry_run),
                overwrite=bool(args.overwrite),
                fill_initial=float(args.fill_initial),
                zero_epsilon=float(args.zero_epsilon),
                candidate_zero_as_invalid=bool(args.candidate_zero_as_invalid),
                resampling=resampling,
                num_threads=int(args.num_threads),
                warp_mem_limit=int(args.warp_mem_limit),
            )

            rows.append(row)

            log(
                "OK",
                f"{city}: status={row['status']}, "
                f"zero_before={row['current_zero_percent_before']}%, "
                f"zero_after={row['current_zero_percent_after']}%, "
                f"repaired={row['repaired_zero_percent_of_current_zero']}%",
            )

        except Exception as exc:
            row = {
                "city": city,
                "status": "failed",
                "candidate_id": candidate.get("candidate_id", ""),
                "candidate_type": candidate.get("candidate_type", ""),
                "candidate_reported_status": candidate.get("status", ""),
                "candidate_fillable_current_zero_percent_reported": candidate.get("fillable_current_zero_percent", ""),
                "candidate_fillable_label_zero_percent_reported": candidate.get("fillable_label_zero_percent", ""),
                "stacked_path": candidate.get("stacked_path", ""),
                "vv_path": candidate.get("vv_path", ""),
                "vh_path": candidate.get("vh_path", ""),
                "current_rtc_path": path_to_str(current_rtc_path(instance_root, city)),
                "label_path": "",
                "output_path": "",
                "backup_path": "",
                "merge_mask_path": "",
                "total_pixels": "",
                "current_zero_pixels_before": "",
                "current_zero_percent_before": "",
                "label_zero_pixels_before": "",
                "repaired_zero_pixels": "",
                "repaired_zero_percent_of_current_zero": "",
                "repaired_label_zero_pixels": "",
                "repaired_label_zero_percent_of_label_zero": "",
                "current_zero_pixels_after": "",
                "current_zero_percent_after": "",
                "label_zero_pixels_after": "",
                "notes": repr(exc),
            }

            rows.append(row)
            log("ERROR", f"{city}: failed with {repr(exc)}")

    summary = build_summary(
        instance_root=instance_root,
        candidate_csv=candidate_csv,
        output_root=output_root,
        rows=rows,
        args=args,
        csv_path=csv_path,
        json_path=json_path,
        md_path=md_path,
    )

    log("STEP", "Writing fallback merge reports.")

    write_csv(csv_path, rows, overwrite=args.overwrite)
    write_json(json_path, summary, overwrite=args.overwrite)
    write_markdown(md_path, summary, rows, overwrite=args.overwrite)

    log("OK", f"Wrote CSV:      {path_to_str(csv_path)}")
    log("OK", f"Wrote JSON:     {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown: {path_to_str(md_path)}")

    log("STEP", "Final summary.")
    log("OK", f"Cities requested: {summary['n_cities_requested']}")
    log("OK", f"Cities completed: {summary['n_cities_completed']}")
    log("OK", f"Cities dry run: {summary['n_cities_dry_run']}")
    log("OK", f"Cities failed: {summary['n_cities_failed']}")

    if summary["n_cities_failed"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()