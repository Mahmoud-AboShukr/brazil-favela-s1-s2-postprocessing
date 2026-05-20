#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
07_search_download_rtc_fallback_products.py

Search, download, and evaluate additional Sentinel-1 RTC fallback products
for RTC zero-coverage repair.

Main purpose:
    - Search Microsoft Planetary Computer Sentinel-1 RTC products.
    - Download VV/VH assets for candidate products.
    - Reproject them to the current Instance C RTC/S2 grid.
    - Measure how much of the current all-zero RTC area they can repair.
    - Output a candidate CSV compatible with 06_merge_rtc_fallback.py.

Important fix:
    This version signs each Planetary Computer asset immediately before download.
    This avoids 403 errors caused by signed URLs expiring during long sequential downloads.

Example for Campo Grande only:

python src/rtc_processing/07_search_download_rtc_fallback_products.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --download-root "D:/my_processed_data/s1_images/rtc_raw_additional" `
  --cities campo_grande `
  --datetime "2020-01-01/2024-12-31" `
  --max-items-per-city 60 `
  --download-top-n 10 `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds
except ImportError as exc:
    raise SystemExit(
        "[ERROR] rasterio is required.\n"
        "Install it with:\n"
        "    pip install rasterio\n\n"
        f"Original error: {exc}"
    )

try:
    from pystac_client import Client
except ImportError as exc:
    raise SystemExit(
        "[ERROR] pystac-client is required.\n"
        "Install it with:\n"
        "    pip install pystac-client\n\n"
        f"Original error: {exc}"
    )

try:
    import planetary_computer as pc
except ImportError as exc:
    raise SystemExit(
        "[ERROR] planetary-computer is required.\n"
        "Install it with:\n"
        "    pip install planetary-computer\n\n"
        f"Original error: {exc}"
    )

try:
    from shapely.geometry import box, shape
except ImportError as exc:
    raise SystemExit(
        "[ERROR] shapely is required.\n"
        "Install it with:\n"
        "    pip install shapely\n\n"
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


def normalize_city(value: str) -> str:
    value = str(value).strip()
    value = value.replace("\\", "/").split("/")[-1]
    value = value.lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

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
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Planetary Computer RTC fallback search")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Download root: `{summary['download_root']}`")
    lines.append(f"- Cities: `{';'.join(summary['cities'])}`")
    lines.append(f"- Datetime: `{summary['parameters']['datetime']}`")
    lines.append(f"- Collection: `{summary['parameters']['collection']}`")
    lines.append(f"- Items found: `{summary['n_items_found']}`")
    lines.append(f"- Items selected for download/evaluation: `{summary['n_items_selected_for_download']}`")
    lines.append(f"- Candidates evaluated: `{summary['n_candidates_evaluated']}`")
    lines.append(f"- Candidates failed: `{summary['n_candidates_failed']}`")
    lines.append("")

    lines.append("## Best candidate per city")
    lines.append("")
    lines.append(
        "| city | current zero % | label-zero overlap % | best item | fillable zero % | "
        "fillable label-zero % | valid % | local VV | local VH |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|---|---|")

    for city in summary["cities"]:
        rows = [
            row for row in candidate_rows
            if row["city"] == city and row["status"] == "ok"
        ]

        rows = sorted(
            rows,
            key=lambda r: (
                -safe_float(r.get("fillable_label_zero_percent", 0.0)),
                -safe_float(r.get("fillable_current_zero_percent", 0.0)),
                r.get("candidate_id", ""),
            ),
        )

        if rows:
            best = rows[0]
            lines.append(
                f"| {city} | "
                f"{best['current_zero_percent']} | "
                f"{best['current_label_zero_overlap_percent']} | "
                f"`{best['item_id']}` | "
                f"{best['fillable_current_zero_percent']} | "
                f"{best['fillable_label_zero_percent']} | "
                f"{best['candidate_valid_percent_on_s2_grid']} | "
                f"`{best['vv_path']}` | "
                f"`{best['vh_path']}` |"
            )
        else:
            lines.append(f"| {city} |  |  | none | 0 | 0 | 0 |  |  |")

    lines.append("")
    lines.append("## Candidate ranking")
    lines.append("")
    lines.append(
        "| city | status | item | datetime | platform | orbit | fillable zero % | "
        "fillable label-zero % | valid % | notes |"
    )
    lines.append("|---|---|---|---|---|---|---:|---:|---:|---|")

    for row in sorted(
        candidate_rows,
        key=lambda r: (
            r.get("city", ""),
            -safe_float(r.get("fillable_label_zero_percent", 0.0)),
            -safe_float(r.get("fillable_current_zero_percent", 0.0)),
            r.get("datetime", ""),
        ),
    ):
        lines.append(
            f"| {row['city']} | "
            f"{row['status']} | "
            f"`{row['item_id']}` | "
            f"{row['datetime']} | "
            f"{row['platform']} | "
            f"{row['orbit_state']} | "
            f"{row['fillable_current_zero_percent']} | "
            f"{row['fillable_label_zero_percent']} | "
            f"{row['candidate_valid_percent_on_s2_grid']} | "
            f"{row['notes']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `fillable zero %` is how much of the current all-zero RTC area the new candidate can replace.")
    lines.append("- `fillable label-zero %` is the most important repair metric because it tells us whether affected favela-positive pixels can be repaired.")
    lines.append("- This script signs each VV/VH asset immediately before download to avoid expired Planetary Computer SAS URLs.")
    lines.append("- If a useful candidate is found, run `06_merge_rtc_fallback.py` using this script's `rtc_fallback_candidate_summary.csv`.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Instance paths
# ---------------------------------------------------------------------

def current_rtc_path(instance_root: Path, city: str) -> Path:
    city = normalize_city(city)
    path = instance_root / "s1_rtc_ready" / city / f"{city}_s1_rtc_vv_vh_10m_aligned.tif"

    if not path.exists():
        fail(f"Current RTC raster missing for {city}: {path_to_str(path)}")

    return path


def label_path(instance_root: Path, city: str) -> Path:
    city = normalize_city(city)
    label_dir = instance_root / "labels" / city

    if not label_dir.exists():
        fail(f"Missing label folder for {city}: {path_to_str(label_dir)}")

    preferred = label_dir / f"{city}_label_final.tif"

    if preferred.exists():
        return preferred

    candidates = sorted(label_dir.glob("*label*.tif"))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        fail(
            f"Ambiguous label raster for {city}:\n"
            + "\n".join(f"  - {path_to_str(p)}" for p in candidates)
        )

    fail(f"Could not find label raster for {city}: {path_to_str(label_dir)}")


# ---------------------------------------------------------------------
# Current masks and search bbox
# ---------------------------------------------------------------------

def masked_data_and_mask(array: np.ma.MaskedArray) -> Tuple[np.ndarray, np.ndarray]:
    data = np.ma.getdata(array)
    mask = np.ma.getmaskarray(array)

    if mask.shape == ():
        mask = np.zeros(data.shape, dtype=bool)

    return data, mask


def read_label_positive(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)

    data, mask = masked_data_and_mask(arr)

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


def compute_current_masks(
    rtc_path: Path,
    label_path_: Path,
    zero_epsilon: float,
) -> Dict[str, object]:
    with rasterio.open(rtc_path) as src:
        rtc = src.read([1, 2]).astype(np.float32)
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height

    label_positive = read_label_positive(label_path_)
    zero = current_zero_mask(rtc, zero_epsilon)
    label_zero = zero & label_positive

    total_pixels = int(zero.size)
    zero_pixels = int(np.count_nonzero(zero))
    label_positive_pixels = int(np.count_nonzero(label_positive))
    label_zero_pixels = int(np.count_nonzero(label_zero))

    return {
        "zero_mask": zero,
        "label_positive_mask": label_positive,
        "label_zero_mask": label_zero,
        "transform": transform,
        "crs": crs,
        "width": width,
        "height": height,
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


def bbox_from_mask(mask: np.ndarray, transform, crs, buffer_pixels: int) -> Tuple[float, float, float, float]:
    rows, cols = np.where(mask)

    if rows.size == 0:
        fail("Cannot build bbox: mask has no true pixels.")

    row_min = max(int(rows.min()) - buffer_pixels, 0)
    row_max = min(int(rows.max()) + buffer_pixels + 1, mask.shape[0])
    col_min = max(int(cols.min()) - buffer_pixels, 0)
    col_max = min(int(cols.max()) + buffer_pixels + 1, mask.shape[1])

    from rasterio.windows import Window, bounds

    window = Window(
        col_off=col_min,
        row_off=row_min,
        width=col_max - col_min,
        height=row_max - row_min,
    )

    left, bottom, right, top = bounds(window, transform)

    if str(crs) != "EPSG:4326":
        left, bottom, right, top = transform_bounds(
            crs,
            "EPSG:4326",
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )

    return (left, bottom, right, top)


# ---------------------------------------------------------------------
# STAC search
# ---------------------------------------------------------------------

def get_asset(item, possible_keys: Sequence[str]):
    for key in possible_keys:
        if key in item.assets:
            return item.assets[key]

    lower_map = {k.lower(): k for k in item.assets.keys()}

    for key in possible_keys:
        lk = key.lower()
        if lk in lower_map:
            return item.assets[lower_map[lk]]

    return None


def item_property(item, keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        if key in item.properties:
            return str(item.properties[key])
    return default


def safe_item_id(item_id: str) -> str:
    text = str(item_id)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def search_pc_items_for_city(
    *,
    bbox_wgs84: Tuple[float, float, float, float],
    datetime_range: str,
    collection: str,
    max_items: int,
) -> List:
    # Important:
    # Do NOT use modifier=pc.sign_inplace here.
    # If we sign during search, URLs may expire before the script reaches later downloads.
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search = client.search(
        collections=[collection],
        bbox=bbox_wgs84,
        datetime=datetime_range,
        limit=max_items,
    )

    items = list(search.items())

    filtered = []

    for item in items:
        vv_asset = get_asset(item, ["vv", "VV"])
        vh_asset = get_asset(item, ["vh", "VH"])

        if vv_asset is None or vh_asset is None:
            continue

        pols = item.properties.get("sar:polarizations") or item.properties.get("s1:polarizations")

        if pols is not None:
            pols_lower = {str(p).lower() for p in pols}
            if "vv" not in pols_lower or "vh" not in pols_lower:
                continue

        filtered.append(item)

    return filtered[:max_items]


def rank_items_by_zero_bbox_overlap(
    items: List,
    bbox_wgs84: Tuple[float, float, float, float],
) -> List[Tuple[object, float]]:
    search_box = box(*bbox_wgs84)
    ranked: List[Tuple[object, float]] = []

    for item in items:
        try:
            geom = shape(item.geometry)
            inter_area = geom.intersection(search_box).area
            denom = search_box.area if search_box.area > 0 else 1.0
            overlap = 100.0 * inter_area / denom
        except Exception:
            overlap = 0.0

        ranked.append((item, overlap))

    ranked.sort(key=lambda x: (-x[1], str(x[0].id)))

    return ranked


# ---------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------

def sign_href_fresh(asset_href: str) -> str:
    """
    Sign one Planetary Computer asset URL immediately before download.

    This is the key fix for the 403 problem. Long downloads can outlive
    SAS URLs that were signed much earlier during STAC search.
    """
    return pc.sign(asset_href)


def download_file(
    url: str,
    output_path: Path,
    *,
    overwrite_downloads: bool,
    timeout: int,
    chunk_size: int = 1024 * 1024,
    show_progress: bool = True,
    description: str = "",
) -> Dict[str, object]:
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite_downloads:
        return {
            "status": "exists",
            "path": path_to_str(output_path),
            "bytes": output_path.stat().st_size,
            "notes": "File already exists; reused local file.",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + f".tmp_{os.getpid()}")

    if tmp_path.exists():
        tmp_path.unlink()

    progress = None

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()

            total_bytes = int(r.headers.get("content-length", 0))
            downloaded_bytes = 0

            if show_progress and tqdm is not None and total_bytes > 0:
                progress = tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=description or output_path.name,
                    leave=True,
                )

            with tmp_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue

                    f.write(chunk)
                    downloaded_bytes += len(chunk)

                    if progress is not None:
                        progress.update(len(chunk))

            if progress is not None:
                progress.close()
                progress = None

        if output_path.exists():
            output_path.unlink()

        tmp_path.replace(output_path)

        return {
            "status": "downloaded",
            "path": path_to_str(output_path),
            "bytes": output_path.stat().st_size,
            "notes": "Downloaded successfully.",
        }

    except Exception as exc:
        if progress is not None:
            progress.close()

        if tmp_path.exists():
            tmp_path.unlink()

        return {
            "status": "failed",
            "path": path_to_str(output_path),
            "bytes": "",
            "notes": repr(exc),
        }


def save_item_json(item, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(item.to_dict(), f, indent=2, ensure_ascii=False)


def download_item_assets(
    item,
    *,
    city: str,
    download_root: Path,
    overwrite_reports: bool,
    overwrite_downloads: bool,
    timeout: int,
    show_progress: bool,
) -> Tuple[Dict[str, object], Dict[str, object], Path, Path, Path]:
    vv_asset = get_asset(item, ["vv", "VV"])
    vh_asset = get_asset(item, ["vh", "VH"])

    if vv_asset is None or vh_asset is None:
        raise ValueError(f"Item {item.id} does not have VV/VH assets. Assets: {list(item.assets.keys())}")

    item_dir = download_root / normalize_city(city) / safe_item_id(item.id)
    vv_path = item_dir / "VV.tif"
    vh_path = item_dir / "VH.tif"
    item_json_path = item_dir / "item.json"

    save_item_json(item, item_json_path, overwrite=overwrite_reports)

    # Important:
    # Fresh signing happens immediately before each actual HTTP download.
    vv_href = sign_href_fresh(vv_asset.href)
    vv_result = download_file(
        vv_href,
        vv_path,
        overwrite_downloads=overwrite_downloads,
        timeout=timeout,
        show_progress=show_progress,
        description=f"{city} {item.id} VV",
    )

    vh_href = sign_href_fresh(vh_asset.href)
    vh_result = download_file(
        vh_href,
        vh_path,
        overwrite_downloads=overwrite_downloads,
        timeout=timeout,
        show_progress=show_progress,
        description=f"{city} {item.id} VH",
    )

    return vv_result, vh_result, vv_path, vh_path, item_json_path


# ---------------------------------------------------------------------
# Reprojection/evaluation
# ---------------------------------------------------------------------

def get_source_nodata(src) -> Optional[float]:
    if src.nodata is None:
        return None

    try:
        return float(src.nodata)
    except Exception:
        return None


def has_normal_georef(src) -> bool:
    if src.crs is None:
        return False

    a = float(src.transform.a)
    e = float(src.transform.e)

    if abs(a - 1.0) < 1e-9 and abs(e - 1.0) < 1e-9:
        return False

    return True


def has_usable_gcps(src) -> bool:
    gcps, gcp_crs = src.gcps
    return bool(gcps and len(gcps) >= 4 and gcp_crs is not None)


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


def evaluate_downloaded_candidate(
    *,
    city: str,
    item,
    vv_path: Path,
    vh_path: Path,
    current_rtc_path_: Path,
    current_masks: Dict[str, object],
    fill_initial: float,
    zero_epsilon: float,
    candidate_zero_as_invalid: bool,
    resampling: Resampling,
    num_threads: int,
    warp_mem_limit: int,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "city": city,
        "candidate_id": f"{city}__pc_rtc__{safe_item_id(item.id)}",
        "candidate_type": "separate_vv_vh",
        "status": "not_started",
        "is_primary_source": 0,
        "item_id": item.id,
        "datetime": item_property(item, ["datetime"]),
        "platform": item_property(item, ["platform"]),
        "orbit_state": item_property(item, ["sat:orbit_state", "s1:orbit_state"]),
        "relative_orbit": item_property(item, ["sat:relative_orbit", "s1:relative_orbit"]),
        "absolute_orbit": item_property(item, ["sat:absolute_orbit", "s1:absolute_orbit"]),
        "stacked_path": "",
        "vv_path": path_to_str(vv_path),
        "vh_path": path_to_str(vh_path),
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
        with rasterio.open(current_rtc_path_) as ref:
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

        valid = candidate_valid_mask(
            vv,
            vh,
            fill_initial=fill_initial,
            zero_epsilon=zero_epsilon,
            candidate_zero_as_invalid=candidate_zero_as_invalid,
        )

        current_zero = current_masks["zero_mask"]
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
# Misc
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


def build_summary(
    *,
    instance_root: Path,
    download_root: Path,
    cities: List[str],
    item_rows: List[Dict[str, object]],
    download_rows: List[Dict[str, object]],
    candidate_rows: List[Dict[str, object]],
    args: argparse.Namespace,
    output_paths: Dict[str, Path],
) -> Dict[str, object]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "download_root": path_to_str(download_root),
        "cities": cities,
        "n_items_found": len(item_rows),
        "n_items_selected_for_download": len(download_rows),
        "n_candidates_evaluated": len(candidate_rows),
        "n_candidates_failed": sum(1 for row in candidate_rows if row["status"] != "ok"),
        "parameters": {
            "collection": args.collection,
            "datetime": args.datetime,
            "max_items_per_city": args.max_items_per_city,
            "download_top_n": args.download_top_n,
            "zero_bbox_buffer_pixels": args.zero_bbox_buffer_pixels,
            "zero_epsilon": args.zero_epsilon,
            "candidate_zero_as_invalid": bool(args.candidate_zero_as_invalid),
            "resampling": args.resampling,
            "fill_initial": args.fill_initial,
            "num_threads": args.num_threads,
            "warp_mem_limit": args.warp_mem_limit,
            "request_timeout": args.request_timeout,
            "overwrite_downloads": bool(args.overwrite_downloads),
        },
        "outputs": {key: path_to_str(value) for key, value in output_paths.items()},
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search/download/evaluate Planetary Computer Sentinel-1 RTC fallback products."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--download-root",
        type=Path,
        required=True,
        help="Where downloaded additional RTC VV/VH assets should be stored.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory. Default: <instance-root>/metadata/rtc_processing/pc_rtc_fallback_search",
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        default=["campo_grande", "sao_goncalo"],
        help="Cities to search. Default: campo_grande sao_goncalo.",
    )

    parser.add_argument(
        "--collection",
        default="sentinel-1-rtc",
        help="Planetary Computer STAC collection. Default: sentinel-1-rtc.",
    )

    parser.add_argument(
        "--datetime",
        default="2021-01-01/2023-12-31",
        help="STAC datetime range. Default: 2021-01-01/2023-12-31.",
    )

    parser.add_argument(
        "--max-items-per-city",
        type=int,
        default=20,
        help="Maximum STAC items fetched per city before ranking. Default: 20.",
    )

    parser.add_argument(
        "--download-top-n",
        type=int,
        default=8,
        help="Download/evaluate top N footprint-ranked candidates per city. Default: 8.",
    )

    parser.add_argument(
        "--zero-bbox-buffer-pixels",
        type=int,
        default=256,
        help="Buffer around current zero mask bbox in pixels before STAC search. Default: 256.",
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
        help="Treat candidate pixels where both VV and VH are zero as invalid. Default: enabled.",
    )

    parser.add_argument(
        "--resampling",
        choices=["nearest", "bilinear", "cubic", "average"],
        default="bilinear",
        help="Resampling for candidate reprojection. Default: bilinear.",
    )

    parser.add_argument(
        "--fill-initial",
        type=float,
        default=-9999.0,
        help="Internal fill value for reprojection. Default: -9999.",
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
        "--request-timeout",
        type=int,
        default=300,
        help="HTTP request timeout in seconds. Default: 300.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite report outputs.",
    )

    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Force redownload even when local VV/VH files already exist.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root
    download_root: Path = args.download_root
    output_dir: Path = args.output_dir or (
        instance_root / "metadata" / "rtc_processing" / "pc_rtc_fallback_search"
    )

    item_csv = output_dir / "pc_rtc_search_items.csv"
    download_csv = output_dir / "pc_rtc_downloads.csv"
    candidate_csv = output_dir / "rtc_fallback_candidate_summary.csv"
    json_path = output_dir / "pc_rtc_fallback_search_report.json"
    md_path = output_dir / "pc_rtc_fallback_search_report.md"

    output_paths = {
        "item_csv": item_csv,
        "download_csv": download_csv,
        "candidate_csv": candidate_csv,
        "json": json_path,
        "markdown": md_path,
    }

    log("STEP", "Searching/downloading additional Sentinel-1 RTC fallback products.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Download root: {path_to_str(download_root)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")
    log("INFO", f"Cities:        {';'.join(args.cities)}")
    log("INFO", f"Datetime:      {args.datetime}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    download_root.mkdir(parents=True, exist_ok=True)

    resampling = resampling_from_name(args.resampling)

    item_rows: List[Dict[str, object]] = []
    download_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []

    cities = [normalize_city(c) for c in args.cities]

    for city in cities:
        log("STEP", f"Processing city: {city}")

        curr_path = current_rtc_path(instance_root, city)
        lab_path = label_path(instance_root, city)

        current_masks = compute_current_masks(
            curr_path,
            lab_path,
            zero_epsilon=float(args.zero_epsilon),
        )

        log(
            "INFO",
            f"{city}: current_zero={current_masks['current_zero_percent']}%, "
            f"label_zero_overlap={current_masks['current_label_zero_overlap_percent']}%",
        )

        if current_masks["current_zero_pixels"] == 0:
            log("WARN", f"{city}: no current zero pixels; skipping.")
            continue

        bbox_wgs84 = bbox_from_mask(
            current_masks["zero_mask"],
            current_masks["transform"],
            current_masks["crs"],
            buffer_pixels=int(args.zero_bbox_buffer_pixels),
        )

        log("INFO", f"{city}: search bbox WGS84={bbox_wgs84}")

        items = search_pc_items_for_city(
            bbox_wgs84=bbox_wgs84,
            datetime_range=args.datetime,
            collection=args.collection,
            max_items=int(args.max_items_per_city),
        )

        ranked = rank_items_by_zero_bbox_overlap(items, bbox_wgs84)

        log("OK", f"{city}: STAC items with VV/VH found: {len(ranked)}")

        for rank, (item, overlap_percent) in enumerate(ranked, start=1):
            item_rows.append(
                {
                    "city": city,
                    "rank": rank,
                    "item_id": item.id,
                    "datetime": item_property(item, ["datetime"]),
                    "platform": item_property(item, ["platform"]),
                    "orbit_state": item_property(item, ["sat:orbit_state", "s1:orbit_state"]),
                    "relative_orbit": item_property(item, ["sat:relative_orbit", "s1:relative_orbit"]),
                    "absolute_orbit": item_property(item, ["sat:absolute_orbit", "s1:absolute_orbit"]),
                    "bbox_overlap_percent_approx": round(float(overlap_percent), 8),
                    "asset_keys": ";".join(item.assets.keys()),
                }
            )

        selected = ranked[: int(args.download_top_n)]

        for rank, (item, overlap_percent) in enumerate(selected, start=1):
            log("STEP", f"{city}: downloading/evaluating item {rank}/{len(selected)}: {item.id}")

            try:
                vv_res, vh_res, vv_path, vh_path, item_json_path = download_item_assets(
                    item,
                    city=city,
                    download_root=download_root,
                    overwrite_reports=bool(args.overwrite),
                    overwrite_downloads=bool(args.overwrite_downloads),
                    timeout=int(args.request_timeout),
                    show_progress=not bool(args.no_progress),
                )

                download_rows.append(
                    {
                        "city": city,
                        "item_id": item.id,
                        "rank": rank,
                        "datetime": item_property(item, ["datetime"]),
                        "platform": item_property(item, ["platform"]),
                        "orbit_state": item_property(item, ["sat:orbit_state", "s1:orbit_state"]),
                        "bbox_overlap_percent_approx": round(float(overlap_percent), 8),
                        "vv_status": vv_res["status"],
                        "vv_path": path_to_str(vv_path),
                        "vv_bytes": vv_res["bytes"],
                        "vv_notes": vv_res["notes"],
                        "vh_status": vh_res["status"],
                        "vh_path": path_to_str(vh_path),
                        "vh_bytes": vh_res["bytes"],
                        "vh_notes": vh_res["notes"],
                        "item_json": path_to_str(item_json_path),
                    }
                )

                if vv_res["status"] == "failed" or vh_res["status"] == "failed":
                    candidate_rows.append(
                        {
                            "city": city,
                            "candidate_id": f"{city}__pc_rtc__{safe_item_id(item.id)}",
                            "candidate_type": "separate_vv_vh",
                            "status": "failed",
                            "is_primary_source": 0,
                            "item_id": item.id,
                            "datetime": item_property(item, ["datetime"]),
                            "platform": item_property(item, ["platform"]),
                            "orbit_state": item_property(item, ["sat:orbit_state", "s1:orbit_state"]),
                            "relative_orbit": item_property(item, ["sat:relative_orbit", "s1:relative_orbit"]),
                            "absolute_orbit": item_property(item, ["sat:absolute_orbit", "s1:absolute_orbit"]),
                            "stacked_path": "",
                            "vv_path": path_to_str(vv_path),
                            "vh_path": path_to_str(vh_path),
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
                            "notes": f"Download failed. VV={vv_res['notes']} VH={vh_res['notes']}",
                        }
                    )
                    continue

                candidate = evaluate_downloaded_candidate(
                    city=city,
                    item=item,
                    vv_path=vv_path,
                    vh_path=vh_path,
                    current_rtc_path_=curr_path,
                    current_masks=current_masks,
                    fill_initial=float(args.fill_initial),
                    zero_epsilon=float(args.zero_epsilon),
                    candidate_zero_as_invalid=bool(args.candidate_zero_as_invalid),
                    resampling=resampling,
                    num_threads=int(args.num_threads),
                    warp_mem_limit=int(args.warp_mem_limit),
                )

                candidate_rows.append(candidate)

                log(
                    "OK" if candidate["status"] == "ok" else "WARN",
                    f"{city}/{item.id}: status={candidate['status']}, "
                    f"fillable_zero={candidate['fillable_current_zero_percent']}%, "
                    f"fillable_label_zero={candidate['fillable_label_zero_percent']}%",
                )

            except Exception as exc:
                log("ERROR", f"{city}/{item.id}: failed with {repr(exc)}")
                candidate_rows.append(
                    {
                        "city": city,
                        "candidate_id": f"{city}__pc_rtc__{safe_item_id(item.id)}",
                        "candidate_type": "separate_vv_vh",
                        "status": "failed",
                        "is_primary_source": 0,
                        "item_id": item.id,
                        "datetime": item_property(item, ["datetime"]),
                        "platform": item_property(item, ["platform"]),
                        "orbit_state": item_property(item, ["sat:orbit_state", "s1:orbit_state"]),
                        "relative_orbit": item_property(item, ["sat:relative_orbit", "s1:relative_orbit"]),
                        "absolute_orbit": item_property(item, ["sat:absolute_orbit", "s1:absolute_orbit"]),
                        "stacked_path": "",
                        "vv_path": "",
                        "vh_path": "",
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
                        "notes": repr(exc),
                    }
                )

    if not item_rows:
        fail("No STAC items found. Try widening --datetime or increasing --zero-bbox-buffer-pixels.")

    if not candidate_rows:
        fail("No candidates were evaluated.")

    summary = build_summary(
        instance_root=instance_root,
        download_root=download_root,
        cities=cities,
        item_rows=item_rows,
        download_rows=download_rows,
        candidate_rows=candidate_rows,
        args=args,
        output_paths=output_paths,
    )

    log("STEP", "Writing reports.")

    write_csv(item_csv, item_rows, overwrite=bool(args.overwrite))
    if download_rows:
        write_csv(download_csv, download_rows, overwrite=bool(args.overwrite))
    write_csv(candidate_csv, candidate_rows, overwrite=bool(args.overwrite))
    write_json(json_path, summary, overwrite=bool(args.overwrite))
    write_markdown(md_path, summary, candidate_rows, overwrite=bool(args.overwrite))

    log("OK", f"Wrote item CSV:      {path_to_str(item_csv)}")
    if download_rows:
        log("OK", f"Wrote download CSV:  {path_to_str(download_csv)}")
    log("OK", f"Wrote candidate CSV: {path_to_str(candidate_csv)}")
    log("OK", f"Wrote JSON:          {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:      {path_to_str(md_path)}")

    log("STEP", "Best candidates by city.")

    for city in cities:
        rows = [
            row for row in candidate_rows
            if row["city"] == city and row["status"] == "ok"
        ]

        rows = sorted(
            rows,
            key=lambda r: (
                -safe_float(r["fillable_label_zero_percent"]),
                -safe_float(r["fillable_current_zero_percent"]),
                r["candidate_id"],
            ),
        )

        if rows:
            best = rows[0]
            log(
                "OK",
                f"{city}: best={best['item_id']}, "
                f"fillable_zero={best['fillable_current_zero_percent']}%, "
                f"fillable_label_zero={best['fillable_label_zero_percent']}%, "
                f"valid={best['candidate_valid_percent_on_s2_grid']}%",
            )
        else:
            log("WARN", f"{city}: no successful candidate.")


if __name__ == "__main__":
    main()