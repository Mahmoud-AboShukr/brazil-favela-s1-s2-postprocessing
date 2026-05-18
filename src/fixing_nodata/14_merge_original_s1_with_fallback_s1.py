#!/usr/bin/env python3
"""
Merge original cropped S1 with processed fallback S1 products.

Purpose:
    Build final S1-ready rasters for instance_C_s2_nodata_repaired.

Inputs:
    instance_C_s2_nodata_repaired/
        s1_snap/<city>/<city>_s1_snap_vv_vh_vvdiff_10m_aligned.tif
        fallback_s1/processed/<city>/<city>_s1_fallback_vv_vh_vvdiff_10m_aligned.tif
        s2_filled/<city>/<city>_s2_12bands_reflectance_10m.tif

Outputs:
    instance_C_s2_nodata_repaired/
        s1_ready/<city>/<city>_s1_ready_vv_vh_vvdiff_10m_aligned.tif

QA:
    qc/s1_ready_merge/
        fill_source/<city>_s1_fill_source.tif
        valid_mask_before/<city>_s1_valid_mask_before_merge.tif
        valid_mask_after/<city>_s1_valid_mask_after_merge.tif
        s1_ready_merge_summary.csv/json/md

Fill-source codes:
    0 = original S1 valid pixel
    1 = filled from processed fallback S1
    2 = filled by nearest-valid-pixel for tiny residual nodata
    9 = unresolved nodata after merge

Merge rule:
    Keep original S1 wherever original S1 is valid.
    Fill only original S1 nodata pixels using fallback S1 where fallback is valid.
    If tiny nodata remains and below threshold, fill with nearest valid pixel.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


FILL_SOURCE_ORIGINAL = 0
FILL_SOURCE_FALLBACK = 1
FILL_SOURCE_NEAREST = 2
FILL_SOURCE_UNRESOLVED = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge original S1 with processed fallback S1 to create s1_ready."
    )

    parser.add_argument(
        "--instance-root",
        type=str,
        required=True,
        help="Root of instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--original-s1-subdir",
        type=str,
        default="s1_snap",
        help="Original S1 subdir. Default: s1_snap",
    )

    parser.add_argument(
        "--fallback-s1-subdir",
        type=str,
        default="fallback_s1/processed",
        help="Processed fallback S1 subdir. Default: fallback_s1/processed",
    )

    parser.add_argument(
        "--s2-subdir",
        type=str,
        default="s2_filled",
        help="S2-filled subdir for alignment checking. Default: s2_filled",
    )

    parser.add_argument(
        "--output-s1-subdir",
        type=str,
        default="s1_ready",
        help="Output final S1-ready subdir. Default: s1_ready",
    )

    parser.add_argument(
        "--qc-subdir",
        type=str,
        default="qc/s1_ready_merge",
        help="QC output subdir. Default: qc/s1_ready_merge",
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional city list. If omitted, all original S1 city folders are processed.",
    )

    parser.add_argument(
        "--original-s1-all-zero-as-nodata",
        action="store_true",
        default=False,
        help="Treat original S1 all-zero-all-band pixels as nodata. Default: False.",
    )

    parser.add_argument(
        "--fallback-s1-all-zero-as-nodata",
        action="store_true",
        default=True,
        help="Treat fallback S1 all-zero-all-band pixels as nodata. Default: True.",
    )

    parser.add_argument(
        "--no-fallback-s1-all-zero-as-nodata",
        dest="fallback_s1_all_zero_as_nodata",
        action="store_false",
        help="Do not treat fallback S1 all-zero-all-band pixels as nodata.",
    )

    parser.add_argument(
        "--nan-as-nodata",
        action="store_true",
        default=True,
        help="Treat NaN/Inf values as nodata. Default: True.",
    )

    parser.add_argument(
        "--fill-tiny-with-nearest",
        action="store_true",
        default=True,
        help="Fill tiny unresolved S1 nodata with nearest valid pixels. Default: True.",
    )

    parser.add_argument(
        "--no-fill-tiny-with-nearest",
        dest="fill_tiny_with_nearest",
        action="store_false",
        help="Disable nearest filling for tiny unresolved S1 nodata.",
    )

    parser.add_argument(
        "--tiny-fill-threshold-percent",
        type=float,
        default=0.5,
        help="Maximum unresolved nodata percent eligible for nearest fill. Default: 0.5.",
    )

    parser.add_argument(
        "--expected-band-count",
        type=int,
        default=3,
        help="Expected S1 band count. Default: 3.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser.parse_args()


def percent(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def safe_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def safe_unlink(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(f"Output exists. Use --overwrite: {path}")


def discover_cities(original_s1_root: Path, cities: list[str] | None) -> list[str]:
    if cities:
        return sorted(cities)

    discovered = sorted([p.name for p in original_s1_root.iterdir() if p.is_dir()])
    if not discovered:
        raise FileNotFoundError(f"No city folders found under {original_s1_root}")

    return discovered


def find_original_s1(instance_root: Path, subdir: str, city: str) -> Path:
    city_dir = instance_root / subdir / city

    candidates = sorted(city_dir.glob(f"{city}_s1_snap_vv_vh_vvdiff_10m_aligned.tif"))
    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s1*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No original S1 found for {city}: {city_dir}")

    return candidates[0]


def find_fallback_s1(instance_root: Path, subdir: str, city: str) -> Path | None:
    city_dir = instance_root / subdir / city

    if not city_dir.exists():
        return None

    candidates = sorted(city_dir.glob(f"{city}_s1_fallback_vv_vh_vvdiff_10m_aligned.tif"))
    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s1_fallback*.tif"))
    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s1*.tif"))

    if not candidates:
        return None

    return candidates[0]


def find_s2_target(instance_root: Path, subdir: str, city: str) -> Path:
    city_dir = instance_root / subdir / city

    candidates = sorted(city_dir.glob(f"{city}_s2_12bands_reflectance_10m.tif"))
    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s2*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No S2-filled target found for {city}: {city_dir}")

    return candidates[0]


def check_same_grid(reference_path: Path, other_path: Path) -> dict:
    with rasterio.open(reference_path) as ref, rasterio.open(other_path) as other:
        same_width = ref.width == other.width
        same_height = ref.height == other.height
        same_crs = ref.crs == other.crs
        same_transform = ref.transform.almost_equals(other.transform)

        return {
            "same_width": same_width,
            "same_height": same_height,
            "same_crs": same_crs,
            "same_transform": same_transform,
            "alignment_ok": same_width and same_height and same_crs and same_transform,
            "reference_width": ref.width,
            "reference_height": ref.height,
            "other_width": other.width,
            "other_height": other.height,
            "reference_crs": str(ref.crs),
            "other_crs": str(other.crs),
        }


def read_s1_data_and_valid_mask(
    path: Path,
    all_zero_as_nodata: bool,
    nan_as_nodata: bool,
    expected_band_count: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
        if src.count != expected_band_count:
            raise ValueError(
                f"Unexpected band count in {path}: {src.count}, expected {expected_band_count}"
            )

        data = src.read(out_dtype="float32")

        masks = src.read_masks()
        official_valid = np.all(masks > 0, axis=0)

        valid = official_valid.copy()

        nonfinite = np.any(~np.isfinite(data), axis=0)
        if nan_as_nodata:
            valid &= ~nonfinite

        all_zero = np.all(data == 0, axis=0)
        if all_zero_as_nodata:
            valid &= ~all_zero

        total = src.width * src.height
        invalid_pixels = int((~valid).sum())

        meta = {
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "nodata_value": src.nodata,
            "total_pixels": int(total),
            "official_invalid_pixels": int((~official_valid).sum()),
            "nonfinite_pixels": int(nonfinite.sum()),
            "all_zero_allbands_pixels": int(all_zero.sum()),
            "combined_invalid_pixels": invalid_pixels,
            "combined_invalid_percent": percent(invalid_pixels, total),
        }

    return data, valid, meta


def nearest_fill_multiband(
    data: np.ndarray,
    valid_mask: np.ndarray,
    fill_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill fill_mask pixels from nearest valid_mask pixels.

    Returns:
        filled_data, newly_filled_mask
    """
    if not fill_mask.any():
        return data, np.zeros_like(fill_mask, dtype=bool)

    if not valid_mask.any():
        raise ValueError("Cannot nearest-fill because there are no valid pixels.")

    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError(
            "scipy is required for nearest filling. Install with: pip install scipy"
        ) from exc

    invalid_for_distance = ~valid_mask

    _, indices = ndimage.distance_transform_edt(
        invalid_for_distance,
        return_distances=True,
        return_indices=True,
    )

    nearest_rows = indices[0]
    nearest_cols = indices[1]

    out = data.copy()
    for band_idx in range(out.shape[0]):
        band = out[band_idx]
        band[fill_mask] = band[
            nearest_rows[fill_mask],
            nearest_cols[fill_mask],
        ]
        out[band_idx] = band

    return out, fill_mask.copy()


def make_s1_output_profile(reference: rasterio.io.DatasetReader) -> dict:
    blockx = min(512, reference.width)
    blocky = min(512, reference.height)

    blockx = max(16, (blockx // 16) * 16)
    blocky = max(16, (blocky // 16) * 16)

    return {
        "driver": "GTiff",
        "height": reference.height,
        "width": reference.width,
        "count": 3,
        "dtype": "float32",
        "crs": reference.crs,
        "transform": reference.transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": blockx,
        "blockysize": blocky,
        "BIGTIFF": "IF_SAFER",
        "nodata": None,
    }


def make_uint8_profile(reference: rasterio.io.DatasetReader) -> dict:
    blockx = min(512, reference.width)
    blocky = min(512, reference.height)

    blockx = max(16, (blockx // 16) * 16)
    blocky = max(16, (blocky // 16) * 16)

    return {
        "driver": "GTiff",
        "height": reference.height,
        "width": reference.width,
        "count": 1,
        "dtype": "uint8",
        "crs": reference.crs,
        "transform": reference.transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": blockx,
        "blockysize": blocky,
        "BIGTIFF": "IF_SAFER",
        "nodata": None,
    }


def write_s1_ready(
    data: np.ndarray,
    valid_mask: np.ndarray,
    reference_path: Path,
    output_path: Path,
    source_tags: dict,
    overwrite: bool,
) -> None:
    safe_unlink(output_path, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as ref:
        profile = make_s1_output_profile(ref)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data.astype(np.float32))
            dst.write_mask(valid_mask.astype(np.uint8) * 255)

            dst.set_band_description(1, "VV_dB")
            dst.set_band_description(2, "VH_dB")
            dst.set_band_description(3, "VV_minus_VH_dB")

            dst.update_tags(**source_tags)


def write_qa_mask(
    array: np.ndarray,
    reference_path: Path,
    output_path: Path,
    description: str,
    overwrite: bool,
) -> None:
    safe_unlink(output_path, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as ref:
        profile = make_uint8_profile(ref)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(array.astype(np.uint8), 1)
            dst.write_mask(np.full(array.shape, 255, dtype=np.uint8))
            dst.set_band_description(1, description)


def inspect_output_nodata(path: Path, nan_as_nodata: bool = True) -> dict:
    with rasterio.open(path) as src:
        data = src.read(out_dtype="float32")
        masks = src.read_masks()

        official_valid = np.all(masks > 0, axis=0)
        valid = official_valid.copy()

        nonfinite = np.any(~np.isfinite(data), axis=0)
        if nan_as_nodata:
            valid &= ~nonfinite

        total = src.width * src.height
        invalid = ~valid

        return {
            "post_merge_invalid_pixels": int(invalid.sum()),
            "post_merge_invalid_percent": percent(int(invalid.sum()), total),
            "post_merge_official_invalid_pixels": int((~official_valid).sum()),
            "post_merge_nonfinite_pixels": int(nonfinite.sum()),
        }


def process_city(
    city: str,
    instance_root: Path,
    original_s1_root: Path,
    fallback_s1_root: Path,
    s2_root: Path,
    output_s1_root: Path,
    qc_root: Path,
    args: argparse.Namespace,
) -> dict:
    original_s1_path = find_original_s1(instance_root, args.original_s1_subdir, city)
    fallback_s1_path = find_fallback_s1(instance_root, args.fallback_s1_subdir, city)
    s2_path = find_s2_target(instance_root, args.s2_subdir, city)

    output_path = (
        output_s1_root
        / city
        / f"{city}_s1_ready_vv_vh_vvdiff_10m_aligned.tif"
    )

    fill_source_path = qc_root / "fill_source" / city / f"{city}_s1_fill_source.tif"
    valid_before_path = qc_root / "valid_mask_before" / city / f"{city}_s1_valid_mask_before_merge.tif"
    valid_after_path = qc_root / "valid_mask_after" / city / f"{city}_s1_valid_mask_after_merge.tif"

    s2_alignment = check_same_grid(original_s1_path, s2_path)
    if not s2_alignment["alignment_ok"]:
        raise ValueError(f"Original S1 and S2_filled are not aligned for {city}")

    original_data, original_valid, original_meta = read_s1_data_and_valid_mask(
        path=original_s1_path,
        all_zero_as_nodata=args.original_s1_all_zero_as_nodata,
        nan_as_nodata=args.nan_as_nodata,
        expected_band_count=args.expected_band_count,
    )

    total_pixels = original_meta["total_pixels"]
    original_missing = ~original_valid

    merged_data = original_data.copy()
    merged_valid = original_valid.copy()

    fill_source = np.zeros(original_valid.shape, dtype=np.uint8)
    fill_source[original_valid] = FILL_SOURCE_ORIGINAL
    fill_source[~original_valid] = FILL_SOURCE_UNRESOLVED

    fallback_used = False
    fallback_fill_pixels = 0
    fallback_invalid_percent = ""
    fallback_path_text = ""

    if fallback_s1_path is not None:
        fallback_path_text = str(fallback_s1_path)
        fallback_alignment = check_same_grid(original_s1_path, fallback_s1_path)

        if not fallback_alignment["alignment_ok"]:
            raise ValueError(f"Original S1 and fallback S1 are not aligned for {city}")

        fallback_data, fallback_valid, fallback_meta = read_s1_data_and_valid_mask(
            path=fallback_s1_path,
            all_zero_as_nodata=args.fallback_s1_all_zero_as_nodata,
            nan_as_nodata=args.nan_as_nodata,
            expected_band_count=args.expected_band_count,
        )

        fallback_invalid_percent = fallback_meta["combined_invalid_percent"]

        fill_from_fallback = original_missing & fallback_valid

        if fill_from_fallback.any():
            merged_data[:, fill_from_fallback] = fallback_data[:, fill_from_fallback]
            merged_valid[fill_from_fallback] = True
            fill_source[fill_from_fallback] = FILL_SOURCE_FALLBACK
            fallback_used = True
            fallback_fill_pixels = int(fill_from_fallback.sum())

    remaining_missing = ~merged_valid
    remaining_missing_percent = percent(int(remaining_missing.sum()), total_pixels)

    nearest_fill_pixels = 0
    nearest_fill_applied = False

    if remaining_missing.any():
        if (
            args.fill_tiny_with_nearest
            and remaining_missing_percent <= args.tiny_fill_threshold_percent
        ):
            merged_data, nearest_filled = nearest_fill_multiband(
                data=merged_data,
                valid_mask=merged_valid,
                fill_mask=remaining_missing,
            )
            merged_valid[nearest_filled] = True
            fill_source[nearest_filled] = FILL_SOURCE_NEAREST
            nearest_fill_pixels = int(nearest_filled.sum())
            nearest_fill_applied = True
        else:
            fill_source[remaining_missing] = FILL_SOURCE_UNRESOLVED

    final_missing = ~merged_valid

    # Recompute VV-minus-VH after merge/fill for consistency.
    merged_data[2] = np.where(
        merged_valid,
        merged_data[0] - merged_data[1],
        0,
    ).astype(np.float32)

    # Set invalid pixels to 0 while preserving dataset mask.
    for b in range(merged_data.shape[0]):
        band = merged_data[b]
        band[~merged_valid] = 0
        merged_data[b] = band

    tags = {
        "source_original_s1": str(original_s1_path),
        "source_fallback_s1": fallback_path_text,
        "merge_rule": "keep_original_valid_fill_original_missing_from_fallback_then_tiny_nearest",
        "fill_source_codes": "0=original,1=fallback,2=nearest,9=unresolved",
        "fallback_used": str(fallback_used),
        "nearest_fill_applied": str(nearest_fill_applied),
    }

    write_s1_ready(
        data=merged_data,
        valid_mask=merged_valid,
        reference_path=original_s1_path,
        output_path=output_path,
        source_tags=tags,
        overwrite=args.overwrite,
    )

    write_qa_mask(
        array=fill_source,
        reference_path=original_s1_path,
        output_path=fill_source_path,
        description="S1 fill source: 0=original, 1=fallback, 2=nearest, 9=unresolved",
        overwrite=args.overwrite,
    )

    write_qa_mask(
        array=original_valid.astype(np.uint8),
        reference_path=original_s1_path,
        output_path=valid_before_path,
        description="S1 valid mask before merge: 1=valid, 0=nodata",
        overwrite=args.overwrite,
    )

    write_qa_mask(
        array=merged_valid.astype(np.uint8),
        reference_path=original_s1_path,
        output_path=valid_after_path,
        description="S1 valid mask after merge: 1=valid, 0=nodata",
        overwrite=args.overwrite,
    )

    post = inspect_output_nodata(output_path, nan_as_nodata=args.nan_as_nodata)

    if post["post_merge_invalid_pixels"] == 0:
        if fallback_used:
            status = "s1_ready_fallback_merged_zero_nodata"
        elif nearest_fill_applied:
            status = "s1_ready_tiny_nearest_filled_zero_nodata"
        else:
            status = "s1_ready_clean_copied_zero_nodata"
    else:
        status = "s1_ready_warning_residual_nodata"

    row = {
        "city": city,
        "status": status,
        "original_s1_path": str(original_s1_path),
        "fallback_s1_path": fallback_path_text,
        "s2_filled_path": str(s2_path),
        "output_path": str(output_path),
        "fill_source_path": str(fill_source_path),
        "valid_mask_before_path": str(valid_before_path),
        "valid_mask_after_path": str(valid_after_path),
        "width": original_meta["width"],
        "height": original_meta["height"],
        "band_count": original_meta["band_count"],
        "total_pixels": total_pixels,
        "original_missing_pixels": int(original_missing.sum()),
        "original_missing_percent": percent(int(original_missing.sum()), total_pixels),
        "fallback_available": fallback_s1_path is not None,
        "fallback_used": fallback_used,
        "fallback_invalid_percent": fallback_invalid_percent,
        "fallback_fill_pixels": fallback_fill_pixels,
        "fallback_fill_percent_of_original_missing": percent(
            fallback_fill_pixels, int(original_missing.sum())
        ),
        "fallback_fill_percent_of_raster": percent(fallback_fill_pixels, total_pixels),
        "nearest_fill_applied": nearest_fill_applied,
        "nearest_fill_pixels": nearest_fill_pixels,
        "nearest_fill_percent_of_raster": percent(nearest_fill_pixels, total_pixels),
        "remaining_missing_before_nearest_pixels": int(remaining_missing.sum()),
        "remaining_missing_before_nearest_percent": remaining_missing_percent,
        "final_missing_pixels": int(final_missing.sum()),
        "final_missing_percent": percent(int(final_missing.sum()), total_pixels),
        "fill_tiny_with_nearest": args.fill_tiny_with_nearest,
        "tiny_fill_threshold_percent": args.tiny_fill_threshold_percent,
        **post,
    }

    return row


def write_json(obj: Any, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(safe_jsonable(obj), f, indent=2, ensure_ascii=False)


def write_csv(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")

    if not rows:
        raise ValueError("No rows to write.")

    fields = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")

    cols = [
        "city",
        "status",
        "original_missing_percent",
        "fallback_available",
        "fallback_fill_percent_of_original_missing",
        "nearest_fill_applied",
        "nearest_fill_pixels",
        "final_missing_percent",
        "post_merge_invalid_percent",
        "output_path",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("# S1-ready merge summary\n\n")
        f.write(
            "This report summarizes the merge of original cropped S1 with processed fallback S1. "
            "Original valid S1 pixels are preserved. Only original S1 nodata pixels are filled.\n\n"
        )

        f.write("## Status counts\n\n")
        counts: dict[str, int] = {}
        for row in rows:
            status = row.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1

        for status, count in sorted(counts.items()):
            f.write(f"- `{status}`: {count}\n")

        f.write("\n## City table\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")

        for row in rows:
            values = []
            for col in cols:
                value = row.get(col, "")
                if isinstance(value, float):
                    value = f"{value:.6f}"
                values.append(str(value).replace("|", "/"))
            f.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()

    instance_root = Path(args.instance_root)

    if not instance_root.exists():
        raise FileNotFoundError(f"Instance root does not exist: {instance_root}")

    original_s1_root = instance_root / args.original_s1_subdir
    fallback_s1_root = instance_root / args.fallback_s1_subdir
    s2_root = instance_root / args.s2_subdir
    output_s1_root = instance_root / args.output_s1_subdir
    qc_root = instance_root / args.qc_subdir

    if not original_s1_root.exists():
        raise FileNotFoundError(f"Original S1 root does not exist: {original_s1_root}")
    if not s2_root.exists():
        raise FileNotFoundError(f"S2-filled root does not exist: {s2_root}")

    output_s1_root.mkdir(parents=True, exist_ok=True)
    qc_root.mkdir(parents=True, exist_ok=True)
    (qc_root / "fill_source").mkdir(parents=True, exist_ok=True)
    (qc_root / "valid_mask_before").mkdir(parents=True, exist_ok=True)
    (qc_root / "valid_mask_after").mkdir(parents=True, exist_ok=True)

    cities = discover_cities(original_s1_root, args.cities)

    print(f"[INFO] Instance root: {instance_root}")
    print(f"[INFO] Original S1 root: {original_s1_root}")
    print(f"[INFO] Fallback S1 root: {fallback_s1_root}")
    print(f"[INFO] S2-filled root: {s2_root}")
    print(f"[INFO] Output S1-ready root: {output_s1_root}")
    print(f"[INFO] QC root: {qc_root}")
    print(f"[INFO] Cities to process: {len(cities)}")
    print(f"[INFO] fill_tiny_with_nearest: {args.fill_tiny_with_nearest}")
    print(f"[INFO] tiny_fill_threshold_percent: {args.tiny_fill_threshold_percent}")

    rows: list[dict] = []

    for idx, city in enumerate(cities, start=1):
        print(f"\n[STEP {idx}/{len(cities)}] {city}")

        try:
            row = process_city(
                city=city,
                instance_root=instance_root,
                original_s1_root=original_s1_root,
                fallback_s1_root=fallback_s1_root,
                s2_root=s2_root,
                output_s1_root=output_s1_root,
                qc_root=qc_root,
                args=args,
            )
            rows.append(row)

            print(
                "[OK] "
                f"status={row['status']} | "
                f"orig_missing={row['original_missing_percent']:.6f}% | "
                f"fallback_fill={row['fallback_fill_percent_of_original_missing']:.6f}% | "
                f"nearest={row['nearest_fill_pixels']} | "
                f"final_missing={row['final_missing_percent']:.6f}%"
            )

        except Exception as exc:
            print(f"[ERROR] {city}: {exc}")
            rows.append(
                {
                    "city": city,
                    "status": "error",
                    "message": str(exc),
                }
            )

    rows = sorted(rows, key=lambda r: str(r["city"]))

    csv_path = qc_root / "s1_ready_merge_summary.csv"
    json_path = qc_root / "s1_ready_merge_summary.json"
    md_path = qc_root / "s1_ready_merge_summary.md"

    write_csv(rows, csv_path, overwrite=args.overwrite)
    write_json(rows, json_path, overwrite=args.overwrite)
    write_markdown(rows, md_path, overwrite=args.overwrite)

    print("\n[DONE] Wrote:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    print("\n[SUMMARY]")
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()