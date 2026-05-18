#!/usr/bin/env python3
"""
Fill residual Sentinel-2 nodata in instance_C_s2_nodata_repaired.

This script repairs only S2 residual nodata after the crop-based repair.

Input:
    instance_C_s2_nodata_repaired/s2/<city>/<city>_s2_12bands_reflectance_10m.tif

Output:
    instance_C_s2_nodata_repaired/s2_filled/<city>/<city>_s2_12bands_reflectance_10m.tif

QA:
    qc/s2_fill/fill_level/<city>_s2_fill_level.tif
    qc/s2_fill/valid_mask_before/<city>_s2_valid_mask_before_fill.tif
    qc/s2_fill/valid_mask_after/<city>_s2_valid_mask_after_fill.tif
    qc/s2_fill/s2_fill_summary.csv/json/md

Fill policy:
    - Do not modify originally valid S2 pixels.
    - Fill only pixels detected as nodata.
    - Fill is nearest-valid-pixel fill, applied independently to each band but using
      the same nodata mask and nearest-pixel indices.
    - This is a conservative spatial fill, not recompositing.
    - QA rasters preserve traceability.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill residual S2 nodata in cropped repaired instance C."
    )

    parser.add_argument(
        "--instance-root",
        type=str,
        required=True,
        help=(
            "Root of instance C, e.g. "
            "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired"
        ),
    )

    parser.add_argument(
        "--input-s2-subdir",
        type=str,
        default="s2",
        help="Input S2 subfolder inside instance root. Default: s2",
    )

    parser.add_argument(
        "--output-s2-subdir",
        type=str,
        default="s2_filled",
        help="Output S2 subfolder inside instance root. Default: s2_filled",
    )

    parser.add_argument(
        "--qc-subdir",
        type=str,
        default="qc/s2_fill",
        help="QC output subfolder inside instance root. Default: qc/s2_fill",
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional city list. If omitted, all cities under input S2 root are processed.",
    )

    s2_zero_group = parser.add_mutually_exclusive_group()
    s2_zero_group.add_argument(
        "--s2-all-zero-as-nodata",
        dest="s2_all_zero_as_nodata",
        action="store_true",
        help="Treat pixels where all S2 bands are exactly zero as nodata.",
    )
    s2_zero_group.add_argument(
        "--no-s2-all-zero-as-nodata",
        dest="s2_all_zero_as_nodata",
        action="store_false",
        help="Do not treat all-zero-all-band S2 pixels as nodata.",
    )
    parser.set_defaults(s2_all_zero_as_nodata=True)

    nan_group = parser.add_mutually_exclusive_group()
    nan_group.add_argument(
        "--nan-as-nodata",
        dest="nan_as_nodata",
        action="store_true",
        help="Treat pixels with NaN/Inf in any S2 band as nodata.",
    )
    nan_group.add_argument(
        "--no-nan-as-nodata",
        dest="nan_as_nodata",
        action="store_false",
        help="Do not treat NaN/Inf as nodata.",
    )
    parser.set_defaults(nan_as_nodata=True)

    parser.add_argument(
        "--fill-code",
        type=int,
        default=3,
        help="Value written to fill_level raster for nearest-neighbor filled pixels. Default: 3",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size for block-wise writing. Default: 512",
    )

    parser.add_argument(
        "--copy-clean",
        action="store_true",
        default=True,
        help=(
            "Copy/write clean S2 cities into output_s2_subdir too, so the filled "
            "S2 folder is complete for all cities. Default: True"
        ),
    )

    parser.add_argument(
        "--skip-verify-output",
        action="store_true",
        help="Skip re-checking nodata after writing output.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing filled S2 and QA rasters.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing outputs.",
    )

    return parser.parse_args()


def percent(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def discover_cities(input_s2_root: Path, cities: list[str] | None) -> list[str]:
    if cities:
        return sorted(cities)

    discovered = sorted([p.name for p in input_s2_root.iterdir() if p.is_dir()])

    if not discovered:
        raise FileNotFoundError(f"No city folders found under {input_s2_root}")

    return discovered


def find_s2_raster(input_s2_root: Path, city: str) -> Path:
    city_dir = input_s2_root / city

    candidates = sorted(city_dir.glob(f"{city}_s2_12bands_reflectance_10m.tif"))

    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s2*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No S2 raster found for {city} under {city_dir}")

    return candidates[0]


def choose_block_size(dim: int, preferred: int = 512) -> int:
    size = min(preferred, dim)
    size = max(16, (size // 16) * 16)
    return size


def iter_windows(height: int, width: int, tile_size: int):
    for row_off in range(0, height, tile_size):
        h = min(tile_size, height - row_off)
        for col_off in range(0, width, tile_size):
            w = min(tile_size, width - col_off)
            yield rasterio.windows.Window(
                col_off=col_off,
                row_off=row_off,
                width=w,
                height=h,
            )


def build_s2_nodata_mask(
    s2_path: Path,
    all_zero_as_nodata: bool,
    nan_as_nodata: bool,
) -> tuple[np.ndarray, dict]:
    """
    Build full-resolution S2 nodata mask.

    combined nodata =
        official raster mask nodata in any band
        OR all-zero-all-band pixels, if requested
        OR non-finite values in any band, if requested
    """
    with rasterio.open(s2_path) as src:
        height = src.height
        width = src.width
        count = src.count
        total_pixels = height * width

        combined = np.zeros((height, width), dtype=bool)
        official = np.zeros((height, width), dtype=bool)
        all_zero = np.zeros((height, width), dtype=bool)
        nonfinite = np.zeros((height, width), dtype=bool)

        band_indexes = list(range(1, count + 1))

        for _, window in src.block_windows(1):
            row0 = int(window.row_off)
            row1 = int(window.row_off + window.height)
            col0 = int(window.col_off)
            col1 = int(window.col_off + window.width)

            masks = src.read_masks(indexes=band_indexes, window=window)
            official_block = np.any(masks == 0, axis=0)
            block_combined = official_block.copy()

            official[row0:row1, col0:col1] = official_block

            if all_zero_as_nodata or nan_as_nodata:
                data = src.read(indexes=band_indexes, window=window)

                if all_zero_as_nodata:
                    all_zero_block = np.all(data == 0, axis=0)
                    all_zero[row0:row1, col0:col1] = all_zero_block
                    block_combined |= all_zero_block

                if nan_as_nodata:
                    nonfinite_block = np.any(~np.isfinite(data), axis=0)
                    nonfinite[row0:row1, col0:col1] = nonfinite_block
                    block_combined |= nonfinite_block

            combined[row0:row1, col0:col1] = block_combined

        meta = {
            "width": width,
            "height": height,
            "band_count": count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "source_nodata_value": src.nodata,
            "total_pixels": int(total_pixels),
            "official_nodata_pixels": int(official.sum()),
            "all_zero_allbands_pixels": int(all_zero.sum()),
            "nonfinite_pixels": int(nonfinite.sum()),
            "combined_nodata_pixels": int(combined.sum()),
            "official_nodata_percent": percent(int(official.sum()), total_pixels),
            "all_zero_allbands_percent": percent(int(all_zero.sum()), total_pixels),
            "nonfinite_percent": percent(int(nonfinite.sum()), total_pixels),
            "combined_nodata_percent": percent(int(combined.sum()), total_pixels),
        }

    return combined, meta


def make_output_profile(src: rasterio.io.DatasetReader) -> dict:
    profile = src.profile.copy()

    blockx = choose_block_size(src.width)
    blocky = choose_block_size(src.height)

    profile.update(
        {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": src.count,
            "dtype": src.dtypes[0],
            "compress": "deflate",
            "predictor": 2 if np.issubdtype(np.dtype(src.dtypes[0]), np.floating) else 1,
            "tiled": True,
            "blockxsize": blockx,
            "blockysize": blocky,
            "BIGTIFF": "IF_SAFER",
            # After filling, there should be no S2 nodata. We rely on the
            # dataset mask instead of nodata=0, because 0 can be a valid numeric value.
            "nodata": None,
        }
    )

    return profile


def make_single_band_uint8_profile(
    reference: rasterio.io.DatasetReader,
    nodata_value: int | None = None,
) -> dict:
    blockx = choose_block_size(reference.width)
    blocky = choose_block_size(reference.height)

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
        "nodata": nodata_value,
    }


def safe_unlink(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")


def write_qa_raster(
    array: np.ndarray,
    reference_path: Path,
    output_path: Path,
    description: str,
    tags: dict,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        return

    safe_unlink(output_path, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as ref:
        profile = make_single_band_uint8_profile(reference=ref, nodata_value=None)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(array.astype(np.uint8), 1)
            dst.write_mask(np.full(array.shape, 255, dtype=np.uint8))
            dst.set_band_description(1, description)
            dst.update_tags(**tags)


def compute_nearest_valid_indices(invalid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute nearest valid pixel indices for every pixel.

    invalid_mask=True means nodata/invalid.
    scipy.ndimage.distance_transform_edt computes nearest zero elements for
    non-zero elements. Therefore, passing invalid_mask gives nearest valid
    coordinates for invalid pixels.
    """
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError(
            "scipy is required for nearest-valid-pixel filling. "
            "Install it with: pip install scipy"
        ) from exc

    if not invalid_mask.any():
        rows = np.indices(invalid_mask.shape)[0]
        cols = np.indices(invalid_mask.shape)[1]
        return rows, cols

    if invalid_mask.all():
        raise ValueError("Cannot fill S2 raster because every pixel is nodata.")

    _, indices = ndimage.distance_transform_edt(
        invalid_mask,
        return_distances=True,
        return_indices=True,
    )

    nearest_rows = indices[0]
    nearest_cols = indices[1]

    return nearest_rows, nearest_cols


def fill_and_write_s2(
    src_path: Path,
    dst_path: Path,
    invalid_mask: np.ndarray,
    fill_code: int,
    overwrite: bool,
    dry_run: bool,
) -> dict:
    if dry_run:
        return {
            "s2_output_path": str(dst_path),
            "s2_written": False,
        }

    safe_unlink(dst_path, overwrite=overwrite)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    fill_needed = bool(invalid_mask.any())

    nearest_rows = None
    nearest_cols = None

    if fill_needed:
        nearest_rows, nearest_cols = compute_nearest_valid_indices(invalid_mask)

    with rasterio.open(src_path) as src:
        profile = make_output_profile(src)

        descriptions = src.descriptions
        source_tags = src.tags()

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                band = src.read(band_idx)

                if fill_needed:
                    band[invalid_mask] = band[
                        nearest_rows[invalid_mask],
                        nearest_cols[invalid_mask],
                    ]

                dst.write(band, band_idx)

                desc = descriptions[band_idx - 1]
                if desc:
                    dst.set_band_description(band_idx, desc)

            # The filled S2 output should have a fully valid dataset mask.
            dst.write_mask(np.full((src.height, src.width), 255, dtype=np.uint8))

            dst.update_tags(**source_tags)
            dst.update_tags(
                s2_fill_applied=str(fill_needed),
                s2_fill_method="nearest_valid_pixel" if fill_needed else "none_needed",
                s2_fill_code=str(fill_code),
                s2_fill_source_path=str(src_path),
            )

    return {
        "s2_output_path": str(dst_path),
        "s2_written": True,
        "fill_needed": fill_needed,
    }


def copy_clean_s2_with_clean_mask(
    src_path: Path,
    dst_path: Path,
    overwrite: bool,
    dry_run: bool,
) -> dict:
    """
    For cities with no S2 nodata, still write a clean copy into s2_filled
    so the folder is complete for downstream processing.
    """
    if dry_run:
        return {
            "s2_output_path": str(dst_path),
            "s2_written": False,
            "fill_needed": False,
        }

    safe_unlink(dst_path, overwrite=overwrite)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        profile = make_output_profile(src)
        descriptions = src.descriptions
        source_tags = src.tags()

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                for window in iter_windows(src.height, src.width, tile_size=512):
                    data = src.read(band_idx, window=window)
                    dst.write(data, band_idx, window=window)

                desc = descriptions[band_idx - 1]
                if desc:
                    dst.set_band_description(band_idx, desc)

            dst.write_mask(np.full((src.height, src.width), 255, dtype=np.uint8))

            dst.update_tags(**source_tags)
            dst.update_tags(
                s2_fill_applied="False",
                s2_fill_method="none_needed",
                s2_fill_source_path=str(src_path),
            )

    return {
        "s2_output_path": str(dst_path),
        "s2_written": True,
        "fill_needed": False,
    }


def verify_output_nodata(
    output_path: Path,
    all_zero_as_nodata: bool,
    nan_as_nodata: bool,
) -> dict:
    mask, meta = build_s2_nodata_mask(
        s2_path=output_path,
        all_zero_as_nodata=all_zero_as_nodata,
        nan_as_nodata=nan_as_nodata,
    )

    return {
        "post_fill_combined_nodata_pixels": int(mask.sum()),
        "post_fill_combined_nodata_percent": meta["combined_nodata_percent"],
        "post_fill_official_nodata_pixels": meta["official_nodata_pixels"],
        "post_fill_all_zero_allbands_pixels": meta["all_zero_allbands_pixels"],
        "post_fill_nonfinite_pixels": meta["nonfinite_pixels"],
    }


def write_csv(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    if not rows:
        raise ValueError("No rows to write.")

    fields = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def write_markdown(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    cols = [
        "city",
        "status",
        "pre_fill_combined_nodata_percent",
        "filled_pixels",
        "filled_pixels_percent",
        "post_fill_combined_nodata_percent",
        "s2_output_path",
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# S2 residual nodata fill summary\n\n")
        f.write(
            "This report summarizes nearest-valid-pixel filling of residual S2 nodata "
            "in instance C. Only S2 rasters are modified; S1 and labels are untouched.\n\n"
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


def process_city(
    city: str,
    input_s2_root: Path,
    output_s2_root: Path,
    qc_root: Path,
    args: argparse.Namespace,
) -> dict:
    src_path = find_s2_raster(input_s2_root, city)
    dst_path = output_s2_root / city / src_path.name

    invalid_mask, meta = build_s2_nodata_mask(
        s2_path=src_path,
        all_zero_as_nodata=args.s2_all_zero_as_nodata,
        nan_as_nodata=args.nan_as_nodata,
    )

    total_pixels = meta["total_pixels"]
    filled_pixels = int(invalid_mask.sum())

    fill_level = np.zeros(invalid_mask.shape, dtype=np.uint8)
    fill_level[invalid_mask] = int(args.fill_code)

    valid_before = (~invalid_mask).astype(np.uint8)
    valid_after = np.ones(invalid_mask.shape, dtype=np.uint8)

    fill_level_path = qc_root / "fill_level" / city / f"{city}_s2_fill_level.tif"
    valid_before_path = qc_root / "valid_mask_before" / city / f"{city}_s2_valid_mask_before_fill.tif"
    valid_after_path = qc_root / "valid_mask_after" / city / f"{city}_s2_valid_mask_after_fill.tif"

    if args.dry_run:
        write_info = {
            "s2_output_path": str(dst_path),
            "s2_written": False,
            "fill_needed": bool(filled_pixels > 0),
        }
    else:
        if filled_pixels > 0:
            write_info = fill_and_write_s2(
                src_path=src_path,
                dst_path=dst_path,
                invalid_mask=invalid_mask,
                fill_code=args.fill_code,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        else:
            if args.copy_clean:
                write_info = copy_clean_s2_with_clean_mask(
                    src_path=src_path,
                    dst_path=dst_path,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
            else:
                write_info = {
                    "s2_output_path": "",
                    "s2_written": False,
                    "fill_needed": False,
                }

        qa_tags = {
            "city": city,
            "source_s2_path": str(src_path),
            "fill_method": "nearest_valid_pixel",
            "fill_code": str(args.fill_code),
        }

        write_qa_raster(
            array=fill_level,
            reference_path=src_path,
            output_path=fill_level_path,
            description="S2 fill level: 0=original valid, 3=nearest-valid fill",
            tags=qa_tags,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

        write_qa_raster(
            array=valid_before,
            reference_path=src_path,
            output_path=valid_before_path,
            description="S2 valid mask before fill: 1=valid, 0=nodata",
            tags=qa_tags,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

        write_qa_raster(
            array=valid_after,
            reference_path=src_path,
            output_path=valid_after_path,
            description="S2 valid mask after fill: 1=valid",
            tags=qa_tags,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    if args.dry_run or args.skip_verify_output or not dst_path.exists():
        verify = {
            "post_fill_combined_nodata_pixels": "",
            "post_fill_combined_nodata_percent": "",
            "post_fill_official_nodata_pixels": "",
            "post_fill_all_zero_allbands_pixels": "",
            "post_fill_nonfinite_pixels": "",
        }
    else:
        verify = verify_output_nodata(
            output_path=dst_path,
            all_zero_as_nodata=args.s2_all_zero_as_nodata,
            nan_as_nodata=args.nan_as_nodata,
        )

    status = "filled" if filled_pixels > 0 else "clean_copied"

    if not args.dry_run and not args.skip_verify_output and dst_path.exists():
        if verify["post_fill_combined_nodata_pixels"] == 0:
            status = status + "_verified_zero_nodata"
        else:
            status = status + "_warning_residual_nodata"

    row = {
        "city": city,
        "status": status,
        "source_s2_path": str(src_path),
        "s2_output_path": str(dst_path),
        "width": meta["width"],
        "height": meta["height"],
        "band_count": meta["band_count"],
        "dtype": meta["dtype"],
        "source_nodata_value": meta["source_nodata_value"],
        "total_pixels": total_pixels,
        "pre_fill_official_nodata_pixels": meta["official_nodata_pixels"],
        "pre_fill_all_zero_allbands_pixels": meta["all_zero_allbands_pixels"],
        "pre_fill_nonfinite_pixels": meta["nonfinite_pixels"],
        "pre_fill_combined_nodata_pixels": meta["combined_nodata_pixels"],
        "pre_fill_official_nodata_percent": meta["official_nodata_percent"],
        "pre_fill_all_zero_allbands_percent": meta["all_zero_allbands_percent"],
        "pre_fill_nonfinite_percent": meta["nonfinite_percent"],
        "pre_fill_combined_nodata_percent": meta["combined_nodata_percent"],
        "filled_pixels": filled_pixels,
        "filled_pixels_percent": percent(filled_pixels, total_pixels),
        "fill_method": "nearest_valid_pixel" if filled_pixels > 0 else "none_needed",
        "fill_code": args.fill_code,
        "fill_level_path": str(fill_level_path),
        "valid_mask_before_path": str(valid_before_path),
        "valid_mask_after_path": str(valid_after_path),
        "dry_run": args.dry_run,
        **write_info,
        **verify,
    }

    return row


def main() -> None:
    args = parse_args()

    instance_root = Path(args.instance_root)

    input_s2_root = instance_root / args.input_s2_subdir
    output_s2_root = instance_root / args.output_s2_subdir
    qc_root = instance_root / args.qc_subdir

    if not instance_root.exists():
        raise FileNotFoundError(f"Instance root does not exist: {instance_root}")

    if not input_s2_root.exists():
        raise FileNotFoundError(f"Input S2 root does not exist: {input_s2_root}")

    if not args.dry_run:
        output_s2_root.mkdir(parents=True, exist_ok=True)
        qc_root.mkdir(parents=True, exist_ok=True)
        (qc_root / "fill_level").mkdir(parents=True, exist_ok=True)
        (qc_root / "valid_mask_before").mkdir(parents=True, exist_ok=True)
        (qc_root / "valid_mask_after").mkdir(parents=True, exist_ok=True)

    cities = discover_cities(input_s2_root, args.cities)

    print(f"[INFO] Instance root: {instance_root}")
    print(f"[INFO] Input S2 root: {input_s2_root}")
    print(f"[INFO] Output S2 root: {output_s2_root}")
    print(f"[INFO] QC root: {qc_root}")
    print(f"[INFO] Cities to process: {len(cities)}")
    print(f"[INFO] S2 all-zero-as-nodata: {args.s2_all_zero_as_nodata}")
    print(f"[INFO] nan-as-nodata: {args.nan_as_nodata}")
    print(f"[INFO] fill_code: {args.fill_code}")
    print(f"[INFO] overwrite: {args.overwrite}")
    print(f"[INFO] dry_run: {args.dry_run}")

    rows: list[dict] = []

    for i, city in enumerate(cities, start=1):
        print(f"\n[STEP {i}/{len(cities)}] {city}")

        try:
            row = process_city(
                city=city,
                input_s2_root=input_s2_root,
                output_s2_root=output_s2_root,
                qc_root=qc_root,
                args=args,
            )
            rows.append(row)

            print(
                "[OK] "
                f"status={row['status']} | "
                f"pre_nodata={row['pre_fill_combined_nodata_percent']:.6f}% | "
                f"filled_pixels={row['filled_pixels']} | "
                f"post_nodata={row['post_fill_combined_nodata_percent']}"
            )

        except Exception as exc:
            print(f"[ERROR] {city}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "city": city,
                    "status": "error",
                    "error": str(exc),
                }
            )

    rows = sorted(rows, key=lambda r: str(r["city"]))

    csv_path = qc_root / "s2_fill_summary.csv"
    json_path = qc_root / "s2_fill_summary.json"
    md_path = qc_root / "s2_fill_summary.md"

    if not args.dry_run:
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