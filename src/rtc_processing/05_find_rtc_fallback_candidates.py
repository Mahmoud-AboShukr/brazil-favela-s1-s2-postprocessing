#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_find_rtc_fallback_candidates.py

Search existing raw RTC data for fallback candidates that could repair
zero-filled / no-coverage areas in the finalized Instance C RTC rasters.

This script DOES NOT merge or modify rasters.

It scans:

    D:/my_processed_data/s1_images/rtc_raw

and focuses on cities that currently have zero-filled RTC coverage in:

    <instance-root>/s1_rtc_ready/<city>/<city>_s1_rtc_vv_vh_10m_aligned.tif

Main idea:

    1. Load the current finalized RTC raster.
    2. Detect current all-zero VV/VH pixels.
    3. Scan raw RTC candidate files for the affected city.
    4. Reproject each candidate to the S2 reference grid.
    5. Estimate how much of the current zero-filled area the candidate could fill.
    6. Report candidate ranking.

Outputs:

    <instance-root>/metadata/rtc_processing/fallback_candidates/
        rtc_fallback_candidate_summary.csv
        rtc_fallback_candidate_files.csv
        rtc_fallback_candidate_report.json
        rtc_fallback_candidate_report.md

Example:

python src/rtc_processing/05_find_rtc_fallback_candidates.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --rtc-raw-root "D:/my_processed_data/s1_images/rtc_raw" `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
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
# Logging
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
            "Output exists and --overwrite was not provided:\n"
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


def normalize_text(value: str) -> str:
    value = str(value).lower().replace("\\", "/")
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_/.]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value


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


# ---------------------------------------------------------------------
# CSV / JSON / Markdown
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
    candidate_rows: List[Dict[str, object]],
    file_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# RTC fallback candidate discovery")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- RTC raw root: `{summary['rtc_raw_root']}`")
    lines.append(f"- Cities searched: `{summary['cities_searched']}`")
    lines.append(f"- Raw raster files found: `{summary['n_raw_raster_files_found']}`")
    lines.append(f"- Candidate sets evaluated: `{summary['n_candidate_sets_evaluated']}`")
    lines.append(f"- Candidate sets failed: `{summary['n_candidate_sets_failed']}`")
    lines.append("")

    lines.append("## Best candidate per city")
    lines.append("")
    lines.append(
        "| city | current all-zero % | label-zero overlap % | best candidate type | "
        "fillable zero % | fillable label-zero % | candidate | notes |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---|---|")

    for city in summary["cities"]:
        city_candidates = [
            row for row in candidate_rows
            if row["city"] == city and row["status"] == "ok"
        ]

        city_candidates = sorted(
            city_candidates,
            key=lambda r: (
                -safe_float(r["fillable_label_zero_percent"]),
                -safe_float(r["fillable_current_zero_percent"]),
                safe_int(r["is_primary_source"], 0),
                r["candidate_id"],
            ),
        )

        if city_candidates:
            best = city_candidates[0]
            lines.append(
                f"| {city} | "
                f"{best['current_zero_percent']} | "
                f"{best['current_label_zero_overlap_percent']} | "
                f"{best['candidate_type']} | "
                f"{best['fillable_current_zero_percent']} | "
                f"{best['fillable_label_zero_percent']} | "
                f"`{best['candidate_id']}` | "
                f"{best['notes']} |"
            )
        else:
            city_info = summary["city_current_zero_stats"].get(city, {})
            lines.append(
                f"| {city} | "
                f"{city_info.get('current_zero_percent', '')} | "
                f"{city_info.get('current_label_zero_overlap_percent', '')} | "
                f"none | 0 | 0 |  | No usable fallback candidate found. |"
            )

    lines.append("")
    lines.append("## Candidate ranking")
    lines.append("")
    lines.append(
        "| city | status | type | primary source? | fillable zero % | fillable label-zero % | "
        "valid % | candidate |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---|")

    for row in sorted(
        candidate_rows,
        key=lambda r: (
            r["city"],
            -safe_float(r.get("fillable_label_zero_percent", 0)),
            -safe_float(r.get("fillable_current_zero_percent", 0)),
            safe_int(r.get("is_primary_source", 0), 0),
        ),
    ):
        lines.append(
            f"| {row['city']} | "
            f"{row['status']} | "
            f"{row['candidate_type']} | "
            f"{row['is_primary_source']} | "
            f"{row['fillable_current_zero_percent']} | "
            f"{row['fillable_label_zero_percent']} | "
            f"{row['candidate_valid_percent_on_s2_grid']} | "
            f"`{row['candidate_id']}` |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `fillable zero %` means the percentage of current RTC all-zero pixels that this candidate could replace with valid VV/VH values.")
    lines.append("- `fillable label-zero %` means the percentage of favela-positive pixels currently affected by zero-filled RTC that this candidate could repair.")
    lines.append("- A good fallback candidate should have high `fillable label-zero %`, especially for Duque de Caxias and São Gonçalo.")
    lines.append("- If no existing candidate can fill the affected label pixels, then we should download/reprocess additional S1/RTC data for that city.")
    lines.append("- This script does not merge anything. The next step would be a fallback merge script if a useful candidate is found.")

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
    "preview",
    "png",
    "jpg",
)


def is_excluded_raster(path: Path) -> bool:
    lower = path.name.lower()
    return any(part in lower for part in EXCLUDE_NAME_PARTS)


def iter_raster_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            if not is_excluded_raster(path):
                yield path


def path_matches_city(path: Path, city: str) -> bool:
    text = normalize_text(path_to_str(path))
    city_norm = normalize_city(city)
    return city_norm in text


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


def inspect_raster_file(path: Path, city: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        "city": city,
        "file_path": path_to_str(path),
        "relative_name": path.name,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        "open_status": "not_started",
        "open_error": "",
        "band_count": "",
        "width": "",
        "height": "",
        "crs": "",
        "has_normal_georef": False,
        "has_gcps": False,
        "gcp_count": 0,
        "gcp_crs": "",
        "role_guess": "",
        "is_candidate_usable": False,
    }

    try:
        with rasterio.open(path) as src:
            row["open_status"] = "ok"
            row["band_count"] = src.count
            row["width"] = src.width
            row["height"] = src.height
            row["crs"] = "" if src.crs is None else str(src.crs)
            row["has_normal_georef"] = has_normal_georef(src)

            gcps, gcp_crs = src.gcps
            row["has_gcps"] = bool(gcps)
            row["gcp_count"] = len(gcps)
            row["gcp_crs"] = "" if gcp_crs is None else str(gcp_crs)

            row["role_guess"] = guess_role_from_name(path, src.count)
            row["is_candidate_usable"] = bool(row["has_normal_georef"] or has_usable_gcps(src))

    except Exception as exc:
        row["open_status"] = "failed"
        row["open_error"] = repr(exc)
        row["role_guess"] = guess_role_from_name(path, None)

    return row


def guess_role_from_name(path: Path, band_count: Optional[int]) -> str:
    text = normalize_text(path_to_str(path))
    stem = normalize_text(path.stem)

    derived_tokens = [
        "vvdiff",
        "vv_diff",
        "vvminusvh",
        "vv_minus_vh",
        "diff",
        "ratio",
    ]

    if any(token in text for token in derived_tokens):
        return "derived_ignore"

    has_vv = bool(re.search(r"(^|[_./])vv($|[_./])", text)) or "vv" in stem
    has_vh = bool(re.search(r"(^|[_./])vh($|[_./])", text)) or "vh" in stem

    if band_count is not None and band_count >= 2:
        if has_vv and has_vh:
            return "stacked_vv_vh"
        if "rtc" in text or "terrain" in text or "gamma" in text or "sigma" in text:
            return "stacked_candidate"

    if has_vv and not has_vh:
        return "vv"

    if has_vh and not has_vv:
        return "vh"

    if has_vv and has_vh:
        return "vv_vh_name_ambiguous"

    if band_count is not None and band_count >= 2:
        return "stacked_candidate"

    return "unknown"


# ---------------------------------------------------------------------
# Instance C raster paths
# ---------------------------------------------------------------------

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
        f"{city}_s2_12bands_reflectance_10m_filled.tif",
        f"{city}_s2_filled_12bands_reflectance_10m.tif",
        f"{city}*s2*12*reflectance*10m*.tif",
        f"{city}*s2*filled*.tif",
        "*s2*12*reflectance*10m*.tif",
        "*s2*filled*.tif",
        "*.tif",
    ]

    for pattern in patterns:
        matches = candidate_tifs(city_dir, [pattern])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            fail(
                f"Ambiguous S2 reference for {city} using pattern {pattern}:\n"
                + "\n".join(f"  - {path_to_str(p)}" for p in matches[:20])
            )

    fail(f"Could not find S2 reference for {city}")


def find_label_reference(instance_root: Path, city: str) -> Path:
    city = normalize_city(city)
    city_dir = instance_root / "labels" / city

    if not city_dir.exists():
        fail(f"Missing label folder for {city}: {path_to_str(city_dir)}")

    patterns = [
        f"{city}_label_final.tif",
        f"{city}*label_final*.tif",
        f"{city}*label*.tif",
        "*label_final*.tif",
        "*label*.tif",
        "*.tif",
    ]

    for pattern in patterns:
        matches = candidate_tifs(city_dir, [pattern])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            fail(
                f"Ambiguous label reference for {city} using pattern {pattern}:\n"
                + "\n".join(f"  - {path_to_str(p)}" for p in matches[:20])
            )

    fail(f"Could not find label reference for {city}")


def find_current_rtc_ready(instance_root: Path, city: str) -> Path:
    city = normalize_city(city)
    path = instance_root / "s1_rtc_ready" / city / f"{city}_s1_rtc_vv_vh_10m_aligned.tif"

    if not path.exists():
        fail(f"Current RTC-ready raster not found for {city}: {path_to_str(path)}")

    return path


# ---------------------------------------------------------------------
# Current zero mask
# ---------------------------------------------------------------------

def masked_data_and_mask(array: np.ma.MaskedArray) -> Tuple[np.ndarray, np.ndarray]:
    data = np.ma.getdata(array)
    mask = np.ma.getmaskarray(array)

    if mask.shape == ():
        mask = np.zeros(data.shape, dtype=bool)

    return data, mask


def compute_current_masks(
    rtc_path: Path,
    label_path: Path,
    zero_epsilon: float,
) -> Dict[str, object]:
    with rasterio.open(rtc_path) as rtc, rasterio.open(label_path) as label:
        rtc_arr = rtc.read([1, 2], masked=True)
        label_arr = label.read(1, masked=True)

    data, mask = masked_data_and_mask(rtc_arr)
    label_data, label_mask = masked_data_and_mask(label_arr)

    finite_both = (
        np.isfinite(data[0])
        & np.isfinite(data[1])
        & ~mask[0]
        & ~mask[1]
    )

    current_zero = (
        finite_both
        & (np.abs(data[0]) <= zero_epsilon)
        & (np.abs(data[1]) <= zero_epsilon)
    )

    label_valid = ~label_mask
    if np.issubdtype(label_data.dtype, np.floating):
        label_valid &= np.isfinite(label_data)

    label_positive = label_valid & (label_data > 0)
    label_zero = label_positive & current_zero

    total_pixels = int(current_zero.size)
    zero_pixels = int(np.count_nonzero(current_zero))
    label_positive_pixels = int(np.count_nonzero(label_positive))
    label_zero_pixels = int(np.count_nonzero(label_zero))

    return {
        "current_zero_mask": current_zero,
        "label_positive_mask": label_positive,
        "label_zero_mask": label_zero,
        "total_pixels": total_pixels,
        "current_zero_pixels": zero_pixels,
        "current_zero_percent": round(100.0 * zero_pixels / total_pixels if total_pixels else 0.0, 8),
        "label_positive_pixels": label_positive_pixels,
        "label_zero_pixels": label_zero_pixels,
        "current_label_zero_overlap_percent": round(
            100.0 * label_zero_pixels / label_positive_pixels
            if label_positive_pixels else 0.0,
            8,
        ),
    }


# ---------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------

def get_source_nodata(src) -> Optional[float]:
    if src.nodata is None:
        return None
    try:
        return float(src.nodata)
    except Exception:
        return None


def reproject_band_to_s2(
    src,
    *,
    band_index: int,
    dst_shape: Tuple[int, int],
    dst_transform,
    dst_crs,
    fill_initial: float,
    resampling: Resampling,
    num_threads: int,
    warp_mem_limit: int,
) -> np.ndarray:
    dst = np.full(dst_shape, fill_initial, dtype=np.float32)
    src_nodata = get_source_nodata(src)

    kwargs = {
        "source": rasterio.band(src, band_index),
        "destination": dst,
        "dst_transform": dst_transform,
        "dst_crs": dst_crs,
        "dst_nodata": fill_initial,
        "resampling": resampling,
        "num_threads": num_threads,
        "warp_mem_limit": warp_mem_limit,
    }

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
        raise ValueError("Source has neither normal georeferencing nor usable GCPs.")

    reproject(**kwargs)

    return dst


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


# ---------------------------------------------------------------------
# Candidate set construction
# ---------------------------------------------------------------------

def scan_city_files(rtc_raw_root: Path, city: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for path in sorted(iter_raster_files(rtc_raw_root)):
        if path_matches_city(path, city):
            rows.append(inspect_raster_file(path, city))

    return rows


def is_primary_source_path(path: str, primary_paths: Sequence[str]) -> bool:
    p = normalize_text(path)
    primary = set(normalize_text(x) for x in primary_paths if x)
    return p in primary


def build_candidate_sets_for_city(
    file_rows: List[Dict[str, object]],
    *,
    city: str,
    primary_paths: Sequence[str],
    max_separate_pairs: int,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []

    usable = [
        r for r in file_rows
        if r["open_status"] == "ok"
        and bool(r["is_candidate_usable"])
        and r["role_guess"] != "derived_ignore"
    ]

    stacked = [
        r for r in usable
        if r["role_guess"] in {"stacked_vv_vh", "stacked_candidate", "vv_vh_name_ambiguous"}
        and safe_int(r["band_count"]) >= 2
    ]

    vv_files = [
        r for r in usable
        if r["role_guess"] == "vv"
    ]

    vh_files = [
        r for r in usable
        if r["role_guess"] == "vh"
    ]

    for idx, row in enumerate(stacked, start=1):
        p = str(row["file_path"])
        candidates.append(
            {
                "city": city,
                "candidate_id": f"{city}__stacked__{idx:03d}",
                "candidate_type": "stacked",
                "stacked_path": p,
                "vv_path": "",
                "vh_path": "",
                "is_primary_source": int(is_primary_source_path(p, primary_paths)),
            }
        )

    # Prefer pairing VV/VH files from the same directory first.
    pair_rows: List[Tuple[Dict[str, object], Dict[str, object], int]] = []

    for vv in vv_files:
        vv_parent = str(Path(str(vv["file_path"])).parent)

        for vh in vh_files:
            vh_parent = str(Path(str(vh["file_path"])).parent)
            same_parent_score = 0 if normalize_text(vv_parent) == normalize_text(vh_parent) else 1
            pair_rows.append((vv, vh, same_parent_score))

    pair_rows = sorted(
        pair_rows,
        key=lambda item: (
            item[2],
            len(str(item[0]["file_path"])) + len(str(item[1]["file_path"])),
            str(item[0]["file_path"]),
            str(item[1]["file_path"]),
        ),
    )

    for idx, (vv, vh, _) in enumerate(pair_rows[:max_separate_pairs], start=1):
        vv_path = str(vv["file_path"])
        vh_path = str(vh["file_path"])
        candidates.append(
            {
                "city": city,
                "candidate_id": f"{city}__separate_vv_vh__{idx:03d}",
                "candidate_type": "separate_vv_vh",
                "stacked_path": "",
                "vv_path": vv_path,
                "vh_path": vh_path,
                "is_primary_source": int(
                    is_primary_source_path(vv_path, primary_paths)
                    and is_primary_source_path(vh_path, primary_paths)
                ),
            }
        )

    return candidates


# ---------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------

def evaluate_candidate(
    candidate: Dict[str, object],
    *,
    s2_path: Path,
    current_masks: Dict[str, object],
    fill_initial: float,
    zero_epsilon: float,
    candidate_zero_as_invalid: bool,
    resampling: Resampling,
    num_threads: int,
    warp_mem_limit: int,
) -> Dict[str, object]:
    city = str(candidate["city"])
    candidate_id = str(candidate["candidate_id"])
    candidate_type = str(candidate["candidate_type"])

    result: Dict[str, object] = {
        "city": city,
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "status": "not_started",
        "is_primary_source": candidate["is_primary_source"],
        "stacked_path": candidate.get("stacked_path", ""),
        "vv_path": candidate.get("vv_path", ""),
        "vh_path": candidate.get("vh_path", ""),
        "candidate_valid_pixels_on_s2_grid": "",
        "candidate_valid_percent_on_s2_grid": "",
        "current_zero_pixels": current_masks["current_zero_pixels"],
        "current_zero_percent": current_masks["current_zero_percent"],
        "current_label_zero_pixels": current_masks["label_zero_pixels"],
        "current_label_zero_overlap_percent": current_masks["current_label_zero_overlap_percent"],
        "fillable_current_zero_pixels": "",
        "fillable_current_zero_percent": "",
        "fillable_label_zero_pixels": "",
        "fillable_label_zero_percent": "",
        "remaining_current_zero_pixels_after_candidate": "",
        "remaining_label_zero_pixels_after_candidate": "",
        "notes": "",
    }

    try:
        with rasterio.open(s2_path) as s2:
            dst_shape = (s2.height, s2.width)
            dst_transform = s2.transform
            dst_crs = s2.crs

            if candidate_type == "stacked":
                stacked_path = Path(str(candidate["stacked_path"]))

                with rasterio.open(stacked_path) as src:
                    if src.count < 2:
                        raise ValueError(f"Stacked candidate has fewer than 2 bands: {src.count}")

                    vv = reproject_band_to_s2(
                        src,
                        band_index=1,
                        dst_shape=dst_shape,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        fill_initial=fill_initial,
                        resampling=resampling,
                        num_threads=num_threads,
                        warp_mem_limit=warp_mem_limit,
                    )

                    vh = reproject_band_to_s2(
                        src,
                        band_index=2,
                        dst_shape=dst_shape,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        fill_initial=fill_initial,
                        resampling=resampling,
                        num_threads=num_threads,
                        warp_mem_limit=warp_mem_limit,
                    )

            elif candidate_type == "separate_vv_vh":
                vv_path = Path(str(candidate["vv_path"]))
                vh_path = Path(str(candidate["vh_path"]))

                with rasterio.open(vv_path) as vv_src, rasterio.open(vh_path) as vh_src:
                    vv = reproject_band_to_s2(
                        vv_src,
                        band_index=1,
                        dst_shape=dst_shape,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        fill_initial=fill_initial,
                        resampling=resampling,
                        num_threads=num_threads,
                        warp_mem_limit=warp_mem_limit,
                    )

                    vh = reproject_band_to_s2(
                        vh_src,
                        band_index=1,
                        dst_shape=dst_shape,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        fill_initial=fill_initial,
                        resampling=resampling,
                        num_threads=num_threads,
                        warp_mem_limit=warp_mem_limit,
                    )

            else:
                raise ValueError(f"Unsupported candidate type: {candidate_type}")

        valid = candidate_valid_mask(
            vv,
            vh,
            fill_initial=fill_initial,
            zero_epsilon=zero_epsilon,
            candidate_zero_as_invalid=candidate_zero_as_invalid,
        )

        current_zero = current_masks["current_zero_mask"]
        label_zero = current_masks["label_zero_mask"]

        candidate_valid_pixels = int(np.count_nonzero(valid))
        total_pixels = int(valid.size)

        fillable_current_zero = valid & current_zero
        fillable_label_zero = valid & label_zero

        fillable_current_zero_pixels = int(np.count_nonzero(fillable_current_zero))
        fillable_label_zero_pixels = int(np.count_nonzero(fillable_label_zero))

        current_zero_pixels = int(current_masks["current_zero_pixels"])
        current_label_zero_pixels = int(current_masks["label_zero_pixels"])

        fillable_current_zero_percent = (
            100.0 * fillable_current_zero_pixels / current_zero_pixels
            if current_zero_pixels else 0.0
        )

        fillable_label_zero_percent = (
            100.0 * fillable_label_zero_pixels / current_label_zero_pixels
            if current_label_zero_pixels else 0.0
        )

        result.update(
            {
                "status": "ok",
                "candidate_valid_pixels_on_s2_grid": candidate_valid_pixels,
                "candidate_valid_percent_on_s2_grid": round(
                    100.0 * candidate_valid_pixels / total_pixels if total_pixels else 0.0,
                    8,
                ),
                "fillable_current_zero_pixels": fillable_current_zero_pixels,
                "fillable_current_zero_percent": round(fillable_current_zero_percent, 8),
                "fillable_label_zero_pixels": fillable_label_zero_pixels,
                "fillable_label_zero_percent": round(fillable_label_zero_percent, 8),
                "remaining_current_zero_pixels_after_candidate": int(current_zero_pixels - fillable_current_zero_pixels),
                "remaining_label_zero_pixels_after_candidate": int(current_label_zero_pixels - fillable_label_zero_pixels),
                "notes": "Candidate evaluated successfully.",
            }
        )

    except Exception as exc:
        result["status"] = "failed"
        result["notes"] = repr(exc)

    return result


# ---------------------------------------------------------------------
# Finalization source paths
# ---------------------------------------------------------------------

def read_primary_sources(finalization_csv: Path) -> Dict[str, List[str]]:
    if not finalization_csv.exists():
        return {}

    rows = read_csv_rows(finalization_csv)
    out: Dict[str, List[str]] = {}

    for row in rows:
        city = normalize_city(row["city"])
        paths = [
            row.get("source_stacked_path", ""),
            row.get("source_vv_path", ""),
            row.get("source_vh_path", ""),
        ]
        out[city] = [p for p in paths if str(p).strip()]

    return out


def affected_cities_from_validation(
    validation_city_csv: Path,
    threshold_percent: float,
) -> List[str]:
    rows = read_csv_rows(validation_city_csv)

    cities = [
        normalize_city(row["city"])
        for row in rows
        if safe_float(row.get("rtc_all_zero_percent", 0.0)) > threshold_percent
    ]

    return sorted(cities)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    *,
    instance_root: Path,
    rtc_raw_root: Path,
    validation_city_csv: Path,
    finalization_csv: Path,
    cities: List[str],
    file_rows: List[Dict[str, object]],
    candidate_rows: List[Dict[str, object]],
    city_zero_stats: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
    output_summary_csv: Path,
    output_files_csv: Path,
    output_json: Path,
    output_md: Path,
) -> Dict[str, object]:
    n_failed = sum(1 for row in candidate_rows if row["status"] != "ok")

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "rtc_raw_root": path_to_str(rtc_raw_root),
        "validation_city_csv": path_to_str(validation_city_csv),
        "finalization_csv": path_to_str(finalization_csv),
        "cities": cities,
        "cities_searched": ";".join(cities),
        "n_raw_raster_files_found": len(file_rows),
        "n_candidate_sets_evaluated": len(candidate_rows),
        "n_candidate_sets_failed": n_failed,
        "city_current_zero_stats": city_zero_stats,
        "parameters": {
            "affected_city_threshold_percent": args.affected_city_threshold_percent,
            "zero_epsilon": args.zero_epsilon,
            "candidate_zero_as_invalid": bool(args.candidate_zero_as_invalid),
            "resampling": args.resampling,
            "fill_initial": args.fill_initial,
            "max_separate_pairs": args.max_separate_pairs,
            "num_threads": args.num_threads,
            "warp_mem_limit": args.warp_mem_limit,
        },
        "outputs": {
            "candidate_summary_csv": path_to_str(output_summary_csv),
            "candidate_files_csv": path_to_str(output_files_csv),
            "json": path_to_str(output_json),
            "markdown": path_to_str(output_md),
        },
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find existing RTC fallback candidates for zero-filled RTC-ready areas."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--rtc-raw-root",
        type=Path,
        required=True,
        help="Root folder containing raw RTC data, e.g. D:/my_processed_data/s1_images/rtc_raw.",
    )

    parser.add_argument(
        "--validation-city-csv",
        type=Path,
        default=None,
        help=(
            "Optional city validation CSV. "
            "Default: <instance-root>/metadata/rtc_processing/s1_rtc_ready_validation_city.csv"
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
            "Default: <instance-root>/metadata/rtc_processing/fallback_candidates"
        ),
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Optional explicit cities to search. Default: auto-detect cities with RTC all-zero percent > threshold.",
    )

    parser.add_argument(
        "--affected-city-threshold-percent",
        type=float,
        default=0.0,
        help="Auto-select cities with RTC all-zero percent greater than this threshold. Default: 0.0.",
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
        help="Treat candidate pixels with both VV and VH equal to zero as invalid. Default: enabled.",
    )

    parser.add_argument(
        "--resampling",
        choices=["nearest", "bilinear", "cubic", "average"],
        default="bilinear",
        help="Resampling method for candidate reprojection. Default: bilinear.",
    )

    parser.add_argument(
        "--fill-initial",
        type=float,
        default=-9999.0,
        help="Internal fill value used during candidate reprojection. Default: -9999.",
    )

    parser.add_argument(
        "--max-separate-pairs",
        type=int,
        default=30,
        help="Maximum VV/VH pair combinations per city. Default: 30.",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="GDAL warp threads per candidate reprojection. Default: 2.",
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
        help="Overwrite outputs.",
    )

    return parser.parse_args()


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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root
    rtc_raw_root: Path = args.rtc_raw_root

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not rtc_raw_root.exists():
        fail(f"RTC raw root does not exist: {path_to_str(rtc_raw_root)}")

    validation_city_csv: Path = args.validation_city_csv or (
        instance_root
        / "metadata"
        / "rtc_processing"
        / "s1_rtc_ready_validation_city.csv"
    )

    finalization_csv: Path = args.finalization_csv or (
        instance_root
        / "metadata"
        / "rtc_processing"
        / "s1_rtc_ready_finalization.csv"
    )

    output_dir: Path = args.output_dir or (
        instance_root
        / "metadata"
        / "rtc_processing"
        / "fallback_candidates"
    )

    output_summary_csv = output_dir / "rtc_fallback_candidate_summary.csv"
    output_files_csv = output_dir / "rtc_fallback_candidate_files.csv"
    output_json = output_dir / "rtc_fallback_candidate_report.json"
    output_md = output_dir / "rtc_fallback_candidate_report.md"

    log("STEP", "Finding RTC fallback candidates.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"RTC raw root:  {path_to_str(rtc_raw_root)}")
    log("INFO", f"Validation CSV: {path_to_str(validation_city_csv)}")
    log("INFO", f"Finalization CSV: {path_to_str(finalization_csv)}")
    log("INFO", f"Output dir: {path_to_str(output_dir)}")

    if args.cities:
        cities = sorted(normalize_city(city) for city in args.cities)
    else:
        cities = affected_cities_from_validation(
            validation_city_csv,
            threshold_percent=args.affected_city_threshold_percent,
        )

    if not cities:
        fail(
            "No affected cities found. Use --cities to provide explicit city names, "
            "or lower --affected-city-threshold-percent."
        )

    log("OK", f"Cities to search: {', '.join(cities)}")

    primary_sources = read_primary_sources(finalization_csv)
    resampling = resampling_from_name(args.resampling)

    all_file_rows: List[Dict[str, object]] = []
    all_candidate_results: List[Dict[str, object]] = []
    city_zero_stats: Dict[str, Dict[str, object]] = {}

    for city in cities:
        log("STEP", f"Processing city: {city}")

        s2_path = find_s2_reference(instance_root, city)
        label_path = find_label_reference(instance_root, city)
        current_rtc_path = find_current_rtc_ready(instance_root, city)

        current_masks = compute_current_masks(
            rtc_path=current_rtc_path,
            label_path=label_path,
            zero_epsilon=args.zero_epsilon,
        )

        city_zero_stats[city] = {
            "current_zero_pixels": current_masks["current_zero_pixels"],
            "current_zero_percent": current_masks["current_zero_percent"],
            "label_positive_pixels": current_masks["label_positive_pixels"],
            "label_zero_pixels": current_masks["label_zero_pixels"],
            "current_label_zero_overlap_percent": current_masks["current_label_zero_overlap_percent"],
        }

        log(
            "INFO",
            f"{city}: current_zero={current_masks['current_zero_percent']}%, "
            f"label_zero_overlap={current_masks['current_label_zero_overlap_percent']}%",
        )

        file_rows = scan_city_files(rtc_raw_root, city)
        all_file_rows.extend(file_rows)

        log("INFO", f"{city}: raw matching raster files found: {len(file_rows)}")

        candidates = build_candidate_sets_for_city(
            file_rows,
            city=city,
            primary_paths=primary_sources.get(city, []),
            max_separate_pairs=args.max_separate_pairs,
        )

        log("INFO", f"{city}: candidate sets built: {len(candidates)}")

        if not candidates:
            all_candidate_results.append(
                {
                    "city": city,
                    "candidate_id": f"{city}__none",
                    "candidate_type": "none",
                    "status": "failed",
                    "is_primary_source": 0,
                    "stacked_path": "",
                    "vv_path": "",
                    "vh_path": "",
                    "candidate_valid_pixels_on_s2_grid": "",
                    "candidate_valid_percent_on_s2_grid": "",
                    "current_zero_pixels": current_masks["current_zero_pixels"],
                    "current_zero_percent": current_masks["current_zero_percent"],
                    "current_label_zero_pixels": current_masks["label_zero_pixels"],
                    "current_label_zero_overlap_percent": current_masks["current_label_zero_overlap_percent"],
                    "fillable_current_zero_pixels": 0,
                    "fillable_current_zero_percent": 0.0,
                    "fillable_label_zero_pixels": 0,
                    "fillable_label_zero_percent": 0.0,
                    "remaining_current_zero_pixels_after_candidate": current_masks["current_zero_pixels"],
                    "remaining_label_zero_pixels_after_candidate": current_masks["label_zero_pixels"],
                    "notes": "No usable candidate set found.",
                }
            )
            continue

        for idx, candidate in enumerate(candidates, start=1):
            log("INFO", f"{city}: evaluating candidate {idx}/{len(candidates)}: {candidate['candidate_id']}")

            result = evaluate_candidate(
                candidate,
                s2_path=s2_path,
                current_masks=current_masks,
                fill_initial=float(args.fill_initial),
                zero_epsilon=float(args.zero_epsilon),
                candidate_zero_as_invalid=bool(args.candidate_zero_as_invalid),
                resampling=resampling,
                num_threads=int(args.num_threads),
                warp_mem_limit=int(args.warp_mem_limit),
            )

            all_candidate_results.append(result)

            log(
                "OK" if result["status"] == "ok" else "WARN",
                f"{result['candidate_id']}: "
                f"status={result['status']}, "
                f"fillable_zero={result['fillable_current_zero_percent']}%, "
                f"fillable_label_zero={result['fillable_label_zero_percent']}%",
            )

    summary = build_summary(
        instance_root=instance_root,
        rtc_raw_root=rtc_raw_root,
        validation_city_csv=validation_city_csv,
        finalization_csv=finalization_csv,
        cities=cities,
        file_rows=all_file_rows,
        candidate_rows=all_candidate_results,
        city_zero_stats=city_zero_stats,
        args=args,
        output_summary_csv=output_summary_csv,
        output_files_csv=output_files_csv,
        output_json=output_json,
        output_md=output_md,
    )

    log("STEP", "Writing fallback candidate reports.")

    write_csv(output_summary_csv, all_candidate_results, overwrite=args.overwrite)
    write_csv(output_files_csv, all_file_rows, overwrite=args.overwrite)
    write_json(output_json, summary, overwrite=args.overwrite)
    write_markdown(output_md, summary, all_candidate_results, all_file_rows, overwrite=args.overwrite)

    log("OK", f"Wrote candidate summary CSV: {path_to_str(output_summary_csv)}")
    log("OK", f"Wrote file inventory CSV:     {path_to_str(output_files_csv)}")
    log("OK", f"Wrote JSON:                   {path_to_str(output_json)}")
    log("OK", f"Wrote Markdown:               {path_to_str(output_md)}")

    log("STEP", "Best candidates by city.")

    for city in cities:
        city_candidates = [
            row for row in all_candidate_results
            if row["city"] == city and row["status"] == "ok"
        ]

        city_candidates = sorted(
            city_candidates,
            key=lambda r: (
                -safe_float(r["fillable_label_zero_percent"]),
                -safe_float(r["fillable_current_zero_percent"]),
                safe_int(r["is_primary_source"], 0),
                r["candidate_id"],
            ),
        )

        if city_candidates:
            best = city_candidates[0]
            log(
                "OK",
                f"{city}: best={best['candidate_id']}, "
                f"fillable_zero={best['fillable_current_zero_percent']}%, "
                f"fillable_label_zero={best['fillable_label_zero_percent']}%, "
                f"primary={best['is_primary_source']}",
            )
        else:
            log("WARN", f"{city}: no successful candidate.")


if __name__ == "__main__":
    main()