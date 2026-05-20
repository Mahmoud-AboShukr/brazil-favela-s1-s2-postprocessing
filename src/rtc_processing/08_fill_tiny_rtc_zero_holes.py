#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
08_fill_tiny_rtc_zero_holes.py

Conservatively fill tiny residual all-zero VV/VH holes in finalized RTC rasters.

This script is intended for tiny residual RTC no-coverage artifacts only,
for example Sorocaba after the major fallback repairs.

It should NOT be used for large missing areas such as the original Campo Grande,
Duque de Caxias, or Sao Goncalo problems.

Logic:
    - Read current s1_rtc_ready/<city>/<city>_s1_rtc_vv_vh_10m_aligned.tif
    - Detect pixels where both VV and VH are zero.
    - Check that all-zero percentage is below a strict threshold.
    - Check that zero pixels do not overlap positive label pixels.
    - Fill zero pixels using nearest valid neighbouring RTC pixels.
    - Write backup before replacement.
    - Write a fill mask for transparency.
    - Write CSV/JSON/Markdown reports.

Default target:
    sorocaba only

Example dry run:

python src/rtc_processing/08_fill_tiny_rtc_zero_holes.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --cities sorocaba `
  --dry-run `
  --overwrite

Example real run:

python src/rtc_processing/08_fill_tiny_rtc_zero_holes.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --cities sorocaba `
  --replace-current `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
except ImportError as exc:
    raise SystemExit(
        "[ERROR] rasterio is required.\n"
        "Install it with:\n"
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


def normalize_city(value: str) -> str:
    value = str(value).strip()
    value = value.replace("\\", "/").split("/")[-1]
    value = value.lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# CSV / JSON / Markdown
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
    rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Tiny RTC zero-hole fill report")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Cities requested: `{summary['n_cities_requested']}`")
    lines.append(f"- Cities completed: `{summary['n_cities_completed']}`")
    lines.append(f"- Cities dry run: `{summary['n_cities_dry_run']}`")
    lines.append(f"- Cities failed: `{summary['n_cities_failed']}`")
    lines.append(f"- Replace current: `{summary['parameters']['replace_current']}`")
    lines.append(f"- Dry run: `{summary['parameters']['dry_run']}`")
    lines.append(f"- Max allowed zero percent: `{summary['parameters']['max_allowed_zero_percent']}`")
    lines.append(f"- Max allowed label-zero overlap percent: `{summary['parameters']['max_allowed_label_zero_overlap_percent']}`")
    lines.append("")

    lines.append("## City-level results")
    lines.append("")
    lines.append(
        "| city | status | zero before % | zero after % | label-zero before % | "
        "filled pixels | iterations | output | notes |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|")

    for row in rows:
        lines.append(
            f"| {row['city']} | "
            f"{row['status']} | "
            f"{row['zero_percent_before']} | "
            f"{row['zero_percent_after']} | "
            f"{row['label_zero_overlap_percent_before']} | "
            f"{row['filled_pixels']} | "
            f"{row['fill_iterations']} | "
            f"`{row['output_path']}` | "
            f"{row['notes']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- This script only fills pixels where both VV and VH are zero.")
    lines.append("- Filling is performed by iterative nearest-neighbour propagation from surrounding valid RTC pixels.")
    lines.append("- Valid non-zero RTC pixels are never changed.")
    lines.append("- The script is intentionally conservative and should only be used for tiny residual holes with no label-positive overlap.")
    lines.append("- After this script, rerun `04_validate_s1_rtc_ready.py`.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def current_rtc_path(instance_root: Path, city: str) -> Path:
    city = normalize_city(city)
    path = instance_root / "s1_rtc_ready" / city / f"{city}_s1_rtc_vv_vh_10m_aligned.tif"

    if not path.exists():
        fail(f"RTC raster does not exist for {city}: {path_to_str(path)}")

    return path


def label_path(instance_root: Path, city: str) -> Path:
    city = normalize_city(city)
    label_dir = instance_root / "labels" / city

    if not label_dir.exists():
        fail(f"Label folder does not exist for {city}: {path_to_str(label_dir)}")

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
# Raster helpers
# ---------------------------------------------------------------------

def read_current_rtc(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    with rasterio.open(path) as src:
        if src.count != 2:
            fail(f"RTC raster must have exactly 2 bands, got {src.count}: {path_to_str(path)}")

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


def zero_mask_rtc(arr: np.ndarray, zero_epsilon: float) -> np.ndarray:
    return (
        np.isfinite(arr[0])
        & np.isfinite(arr[1])
        & (np.abs(arr[0]) <= zero_epsilon)
        & (np.abs(arr[1]) <= zero_epsilon)
    )


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def grids_match(rtc_path: Path, label_path_: Path, tolerance: float = 0.0) -> bool:
    with rasterio.open(rtc_path) as rtc, rasterio.open(label_path_) as label:
        if rtc.width != label.width or rtc.height != label.height:
            return False

        if rtc.crs != label.crs:
            return False

        a = (
            rtc.transform.a,
            rtc.transform.b,
            rtc.transform.c,
            rtc.transform.d,
            rtc.transform.e,
            rtc.transform.f,
        )

        b = (
            label.transform.a,
            label.transform.b,
            label.transform.c,
            label.transform.d,
            label.transform.e,
            label.transform.f,
        )

        return all(abs(float(x) - float(y)) <= tolerance for x, y in zip(a, b))


# ---------------------------------------------------------------------
# Nearest-neighbour propagation fill
# ---------------------------------------------------------------------

NEIGHBOUR_OFFSETS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


def shifted_view(arr: np.ndarray, dy: int, dx: int, fill_value: float = 0.0) -> np.ndarray:
    """
    Return arr shifted by dy/dx.

    The returned pixel [r, c] contains source value [r + dy, c + dx].
    This is useful for asking: "what does my neighbour contain?"
    """

    out = np.full(arr.shape, fill_value, dtype=arr.dtype)

    h, w = arr.shape

    src_r0 = max(0, dy)
    src_r1 = min(h, h + dy)
    dst_r0 = max(0, -dy)
    dst_r1 = min(h, h - dy)

    src_c0 = max(0, dx)
    src_c1 = min(w, w + dx)
    dst_c0 = max(0, -dx)
    dst_c1 = min(w, w - dx)

    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return out

    out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]

    return out


def fill_zero_holes_nearest(
    arr: np.ndarray,
    zero_mask: np.ndarray,
    *,
    max_iterations: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fill zero-mask pixels by iterative nearest-neighbour propagation.

    This is a conservative raster-only fill:
        - The invalid mask is the all-zero VV/VH mask.
        - At each iteration, invalid pixels adjacent to any valid pixel are filled.
        - VV and VH are copied together from the same neighbouring pixel.
        - Valid pixels are never changed.

    Returns:
        filled_array
        actually_filled_mask
        iterations_used
    """

    if arr.shape[0] != 2:
        raise ValueError(f"Expected RTC array shape (2, H, W), got {arr.shape}")

    filled = arr.copy()

    remaining = zero_mask.copy()
    actually_filled = np.zeros(zero_mask.shape, dtype=bool)

    valid = ~remaining & np.isfinite(filled[0]) & np.isfinite(filled[1])

    if not np.any(valid):
        raise ValueError("No valid RTC pixels available for nearest-neighbour fill.")

    iterations_used = 0

    for iteration in range(1, max_iterations + 1):
        if not np.any(remaining):
            break

        filled_this_iteration = np.zeros(remaining.shape, dtype=bool)

        for dy, dx in NEIGHBOUR_OFFSETS:
            neighbour_valid = shifted_view(valid.astype(np.uint8), dy, dx, fill_value=0).astype(bool)

            targets = remaining & neighbour_valid & ~filled_this_iteration

            if not np.any(targets):
                continue

            neighbour_vv = shifted_view(filled[0], dy, dx, fill_value=0.0)
            neighbour_vh = shifted_view(filled[1], dy, dx, fill_value=0.0)

            filled[0][targets] = neighbour_vv[targets]
            filled[1][targets] = neighbour_vh[targets]

            filled_this_iteration[targets] = True

        if not np.any(filled_this_iteration):
            raise ValueError(
                "Nearest fill stalled before all zero pixels were filled. "
                "This should not happen unless the raster has isolated invalid regions with no valid boundary."
            )

        actually_filled |= filled_this_iteration
        remaining[filled_this_iteration] = False
        valid |= filled_this_iteration

        iterations_used = iteration

    if np.any(remaining):
        raise ValueError(
            f"Nearest fill reached max_iterations={max_iterations} with "
            f"{int(np.count_nonzero(remaining))} zero pixels still unfilled."
        )

    return filled, actually_filled, iterations_used


# ---------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------

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


def write_rtc_output(
    output_path: Path,
    *,
    arr: np.ndarray,
    meta: Dict[str, object],
    city: str,
    filled_pixels: int,
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(output_path, overwrite)

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

    tmp_path = output_path.with_name(output_path.stem + f".tmp_{os.getpid()}" + output_path.suffix)

    if tmp_path.exists():
        tmp_path.unlink()

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
            dst.update_tags(
                tiny_zero_fill="true",
                tiny_zero_fill_city=city,
                tiny_zero_fill_method="iterative_nearest_neighbour",
                tiny_zero_fill_pixels=str(filled_pixels),
                tiny_zero_fill_created_utc=datetime.now(timezone.utc).isoformat(),
            )

        if output_path.exists():
            output_path.unlink()

        tmp_path.replace(output_path)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_fill_mask(
    output_path: Path,
    *,
    mask: np.ndarray,
    ref_path: Path,
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(output_path, overwrite)

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

    tmp_path = output_path.with_name(output_path.stem + f".tmp_{os.getpid()}" + output_path.suffix)

    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
            dst.set_band_description(1, "tiny_zero_fill_mask")
            dst.update_tags(
                meaning="1 where tiny RTC all-zero VV/VH pixels were filled",
                method="iterative_nearest_neighbour",
                created_utc=datetime.now(timezone.utc).isoformat(),
            )

        if output_path.exists():
            output_path.unlink()

        tmp_path.replace(output_path)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ---------------------------------------------------------------------
# Per-city processing
# ---------------------------------------------------------------------

def process_city(
    city: str,
    *,
    instance_root: Path,
    output_root: Path,
    backup_root: Path,
    mask_root: Path,
    replace_current: bool,
    dry_run: bool,
    overwrite: bool,
    zero_epsilon: float,
    max_allowed_zero_percent: float,
    max_allowed_label_zero_overlap_percent: float,
    max_iterations: int,
) -> Dict[str, object]:
    city = normalize_city(city)

    rtc_path = current_rtc_path(instance_root, city)
    lab_path = label_path(instance_root, city)

    if not grids_match(rtc_path, lab_path):
        raise ValueError("RTC and label grids do not match.")

    arr, meta = read_current_rtc(rtc_path)
    label_positive = read_label_positive(lab_path)

    zero_before = zero_mask_rtc(arr, zero_epsilon)

    total_pixels = int(zero_before.size)
    zero_pixels_before = int(np.count_nonzero(zero_before))
    zero_percent_before = percent(zero_pixels_before, total_pixels)

    label_positive_pixels = int(np.count_nonzero(label_positive))
    label_zero_pixels_before = int(np.count_nonzero(zero_before & label_positive))
    label_zero_overlap_percent_before = percent(label_zero_pixels_before, label_positive_pixels)

    if zero_pixels_before == 0:
        return {
            "city": city,
            "status": "skipped_no_zero_pixels",
            "rtc_path": path_to_str(rtc_path),
            "label_path": path_to_str(lab_path),
            "output_path": path_to_str(rtc_path if replace_current else output_root / city / rtc_path.name),
            "backup_path": "",
            "fill_mask_path": "",
            "total_pixels": total_pixels,
            "zero_pixels_before": zero_pixels_before,
            "zero_percent_before": round(zero_percent_before, 8),
            "label_positive_pixels": label_positive_pixels,
            "label_zero_pixels_before": label_zero_pixels_before,
            "label_zero_overlap_percent_before": round(label_zero_overlap_percent_before, 8),
            "filled_pixels": 0,
            "zero_pixels_after": 0,
            "zero_percent_after": 0.0,
            "label_zero_pixels_after": 0,
            "label_zero_overlap_percent_after": 0.0,
            "fill_iterations": 0,
            "notes": "No all-zero RTC pixels found.",
        }

    if zero_percent_before > max_allowed_zero_percent:
        raise ValueError(
            f"Zero percent before fill is {zero_percent_before:.8f}%, "
            f"which exceeds max_allowed_zero_percent={max_allowed_zero_percent}."
        )

    if label_zero_overlap_percent_before > max_allowed_label_zero_overlap_percent:
        raise ValueError(
            f"Label-positive zero overlap before fill is {label_zero_overlap_percent_before:.8f}%, "
            f"which exceeds max_allowed_label_zero_overlap_percent={max_allowed_label_zero_overlap_percent}."
        )

    filled_arr, filled_mask, iterations_used = fill_zero_holes_nearest(
        arr,
        zero_before,
        max_iterations=max_iterations,
    )

    zero_after = zero_mask_rtc(filled_arr, zero_epsilon)

    zero_pixels_after = int(np.count_nonzero(zero_after))
    zero_percent_after = percent(zero_pixels_after, total_pixels)

    label_zero_pixels_after = int(np.count_nonzero(zero_after & label_positive))
    label_zero_overlap_percent_after = percent(label_zero_pixels_after, label_positive_pixels)

    filled_pixels = int(np.count_nonzero(filled_mask))

    if replace_current:
        output_path = rtc_path
    else:
        output_path = output_root / city / rtc_path.name

    mask_path = mask_root / city / f"{city}_tiny_zero_fill_mask.tif"

    backup_path = ""

    status = "dry_run" if dry_run else "completed"

    if not dry_run:
        if replace_current:
            backup_path_obj = backup_current_file(
                current_path=rtc_path,
                backup_root=backup_root,
                city=city,
                overwrite=overwrite,
            )
            backup_path = path_to_str(backup_path_obj)

        write_rtc_output(
            output_path,
            arr=filled_arr,
            meta=meta,
            city=city,
            filled_pixels=filled_pixels,
            overwrite=overwrite,
        )

        write_fill_mask(
            mask_path,
            mask=filled_mask,
            ref_path=output_path,
            overwrite=overwrite,
        )

    return {
        "city": city,
        "status": status,
        "rtc_path": path_to_str(rtc_path),
        "label_path": path_to_str(lab_path),
        "output_path": path_to_str(output_path),
        "backup_path": backup_path,
        "fill_mask_path": path_to_str(mask_path),
        "total_pixels": total_pixels,
        "zero_pixels_before": zero_pixels_before,
        "zero_percent_before": round(zero_percent_before, 8),
        "label_positive_pixels": label_positive_pixels,
        "label_zero_pixels_before": label_zero_pixels_before,
        "label_zero_overlap_percent_before": round(label_zero_overlap_percent_before, 8),
        "filled_pixels": filled_pixels,
        "zero_pixels_after": zero_pixels_after,
        "zero_percent_after": round(zero_percent_after, 8),
        "label_zero_pixels_after": label_zero_pixels_after,
        "label_zero_overlap_percent_after": round(label_zero_overlap_percent_after, 8),
        "fill_iterations": iterations_used,
        "notes": "Dry run only; no raster written." if dry_run else "Tiny all-zero RTC holes filled with nearest valid neighbours.",
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    *,
    instance_root: Path,
    rows: List[Dict[str, object]],
    args: argparse.Namespace,
    csv_path: Path,
    json_path: Path,
    md_path: Path,
) -> Dict[str, object]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "n_cities_requested": len(rows),
        "n_cities_completed": sum(1 for r in rows if r["status"] == "completed"),
        "n_cities_dry_run": sum(1 for r in rows if r["status"] == "dry_run"),
        "n_cities_failed": sum(1 for r in rows if r["status"] == "failed"),
        "parameters": {
            "cities": [normalize_city(c) for c in args.cities],
            "replace_current": bool(args.replace_current),
            "dry_run": bool(args.dry_run),
            "zero_epsilon": args.zero_epsilon,
            "max_allowed_zero_percent": args.max_allowed_zero_percent,
            "max_allowed_label_zero_overlap_percent": args.max_allowed_label_zero_overlap_percent,
            "max_iterations": args.max_iterations,
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
        description="Conservatively fill tiny residual all-zero RTC VV/VH holes."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        default=["sorocaba"],
        help="Cities to process. Default: sorocaba.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root if --replace-current is not used. Default: <instance-root>/s1_rtc_ready_tiny_zero_filled",
    )

    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help="Backup root if --replace-current is used. Default: <instance-root>/metadata/rtc_processing/backups/s1_rtc_ready_before_tiny_zero_fill",
    )

    parser.add_argument(
        "--mask-root",
        type=Path,
        default=None,
        help="Fill mask root. Default: <instance-root>/metadata/rtc_processing/tiny_zero_fill_masks",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Report directory. Default: <instance-root>/metadata/rtc_processing/tiny_zero_fill",
    )

    parser.add_argument(
        "--zero-epsilon",
        type=float,
        default=1e-6,
        help="Tolerance for treating VV/VH as zero. Default: 1e-6.",
    )

    parser.add_argument(
        "--max-allowed-zero-percent",
        type=float,
        default=1.0,
        help="Maximum allowed all-zero RTC percent before fill. Default: 1.0.",
    )

    parser.add_argument(
        "--max-allowed-label-zero-overlap-percent",
        type=float,
        default=0.0,
        help="Maximum allowed label-positive overlap percent before fill. Default: 0.0.",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2048,
        help="Maximum neighbour-propagation iterations. Default: 2048.",
    )

    parser.add_argument(
        "--replace-current",
        action="store_true",
        help="Replace files inside s1_rtc_ready after creating a backup.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full fill computation but do not write rasters.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite reports/backups/outputs if they already exist.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root

    output_root: Path = args.output_root or (
        instance_root / "s1_rtc_ready_tiny_zero_filled"
    )

    backup_root: Path = args.backup_root or (
        instance_root
        / "metadata"
        / "rtc_processing"
        / "backups"
        / "s1_rtc_ready_before_tiny_zero_fill"
    )

    mask_root: Path = args.mask_root or (
        instance_root
        / "metadata"
        / "rtc_processing"
        / "tiny_zero_fill_masks"
    )

    report_dir: Path = args.report_dir or (
        instance_root
        / "metadata"
        / "rtc_processing"
        / "tiny_zero_fill"
    )

    csv_path = report_dir / "tiny_rtc_zero_fill_summary.csv"
    json_path = report_dir / "tiny_rtc_zero_fill_report.json"
    md_path = report_dir / "tiny_rtc_zero_fill_report.md"

    log("STEP", "Filling tiny RTC all-zero holes.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Cities:        {';'.join(args.cities)}")
    log("INFO", f"Replace current: {args.replace_current}")
    log("INFO", f"Dry run: {args.dry_run}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    rows: List[Dict[str, object]] = []

    for city_raw in args.cities:
        city = normalize_city(city_raw)
        log("STEP", f"Processing city: {city}")

        try:
            row = process_city(
                city,
                instance_root=instance_root,
                output_root=output_root,
                backup_root=backup_root,
                mask_root=mask_root,
                replace_current=bool(args.replace_current),
                dry_run=bool(args.dry_run),
                overwrite=bool(args.overwrite),
                zero_epsilon=float(args.zero_epsilon),
                max_allowed_zero_percent=float(args.max_allowed_zero_percent),
                max_allowed_label_zero_overlap_percent=float(args.max_allowed_label_zero_overlap_percent),
                max_iterations=int(args.max_iterations),
            )

            rows.append(row)

            log(
                "OK",
                f"{city}: status={row['status']}, "
                f"zero_before={row['zero_percent_before']}%, "
                f"zero_after={row['zero_percent_after']}%, "
                f"filled_pixels={row['filled_pixels']}, "
                f"iterations={row['fill_iterations']}",
            )

        except Exception as exc:
            row = {
                "city": city,
                "status": "failed",
                "rtc_path": path_to_str(current_rtc_path(instance_root, city)) if (instance_root / "s1_rtc_ready" / city).exists() else "",
                "label_path": "",
                "output_path": "",
                "backup_path": "",
                "fill_mask_path": "",
                "total_pixels": "",
                "zero_pixels_before": "",
                "zero_percent_before": "",
                "label_positive_pixels": "",
                "label_zero_pixels_before": "",
                "label_zero_overlap_percent_before": "",
                "filled_pixels": "",
                "zero_pixels_after": "",
                "zero_percent_after": "",
                "label_zero_pixels_after": "",
                "label_zero_overlap_percent_after": "",
                "fill_iterations": "",
                "notes": repr(exc),
            }

            rows.append(row)
            log("ERROR", f"{city}: failed with {repr(exc)}")

    summary = build_summary(
        instance_root=instance_root,
        rows=rows,
        args=args,
        csv_path=csv_path,
        json_path=json_path,
        md_path=md_path,
    )

    log("STEP", "Writing reports.")

    write_csv(csv_path, rows, overwrite=bool(args.overwrite))
    write_json(json_path, summary, overwrite=bool(args.overwrite))
    write_markdown(md_path, summary, rows, overwrite=bool(args.overwrite))

    log("OK", f"Wrote CSV:      {path_to_str(csv_path)}")
    log("OK", f"Wrote JSON:     {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown: {path_to_str(md_path)}")

    log("STEP", "Final summary.")
    log("OK", f"Cities completed: {summary['n_cities_completed']}")
    log("OK", f"Cities dry run: {summary['n_cities_dry_run']}")
    log("OK", f"Cities failed: {summary['n_cities_failed']}")

    if summary["n_cities_failed"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()