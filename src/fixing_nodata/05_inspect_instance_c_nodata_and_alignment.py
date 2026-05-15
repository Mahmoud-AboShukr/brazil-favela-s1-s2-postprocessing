#!/usr/bin/env python3
"""
Inspect nodata and alignment in instance_C_s2_nodata_repaired.

This script validates the cropped dataset instance created by:

    04_apply_valid_crop_to_s2_s1_labels.py

It checks, per city:
- S2/S1/label files exist
- expected band counts
- S2/S1/label alignment
- label binary values
- S2 nodata percentage
- S1 nodata percentage
- border-connected vs internal residual nodata

It does not modify any raster.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect nodata and alignment in cropped repaired instance C."
    )

    parser.add_argument(
        "--instance-root",
        type=str,
        required=True,
        help=(
            "Root of repaired instance, e.g. "
            "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "QC output directory. If omitted, outputs are written to "
            "<instance-root>/qc."
        ),
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional list of city slugs. If omitted, all cities under instance S2 root are processed.",
    )

    zero_group = parser.add_mutually_exclusive_group()
    zero_group.add_argument(
        "--s2-all-zero-as-nodata",
        dest="s2_all_zero_as_nodata",
        action="store_true",
        help="Treat S2 pixels where all bands are exactly zero as nodata.",
    )
    zero_group.add_argument(
        "--no-s2-all-zero-as-nodata",
        dest="s2_all_zero_as_nodata",
        action="store_false",
        help="Do not treat all-zero S2 pixels as nodata.",
    )
    parser.set_defaults(s2_all_zero_as_nodata=True)

    s1_zero_group = parser.add_mutually_exclusive_group()
    s1_zero_group.add_argument(
        "--s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_true",
        help="Treat S1 pixels where all bands are exactly zero as nodata.",
    )
    s1_zero_group.add_argument(
        "--no-s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_false",
        help="Do not treat all-zero S1 pixels as nodata.",
    )
    parser.set_defaults(s1_all_zero_as_nodata=False)

    parser.add_argument(
        "--nan-as-nodata",
        action="store_true",
        default=True,
        help="Treat NaN/Inf values as nodata.",
    )

    parser.add_argument(
        "--border-margin",
        type=int,
        default=128,
        help="Border margin in pixels for border-connected nodata classification.",
    )

    parser.add_argument(
        "--expected-s2-bands",
        type=int,
        default=12,
        help="Expected number of S2 bands.",
    )

    parser.add_argument(
        "--expected-s1-bands",
        type=int,
        default=3,
        help="Expected number of S1 bands.",
    )

    parser.add_argument(
        "--expected-label-bands",
        type=int,
        default=1,
        help="Expected number of label bands.",
    )

    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.5,
        help="Label pixels > this value are treated as positive.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing QC outputs.",
    )

    return parser.parse_args()


def percent(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def discover_cities(s2_root: Path, cities: list[str] | None) -> list[str]:
    if cities:
        return sorted(cities)

    discovered = []

    for path in sorted(s2_root.iterdir()):
        if path.is_dir():
            discovered.append(path.name)

    if not discovered:
        raise FileNotFoundError(f"No city folders found under {s2_root}")

    return discovered


def find_s2_raster(instance_root: Path, city: str) -> Path:
    root = instance_root / "s2"
    candidates = sorted((root / city).glob(f"{city}_s2_12bands_reflectance_10m.tif"))

    if not candidates:
        candidates = sorted((root / city).glob(f"{city}_s2*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No S2 raster found for {city} under {root}")

    return candidates[0]


def find_s1_raster(instance_root: Path, city: str) -> Path:
    root = instance_root / "s1_snap"
    candidates = sorted((root / city).glob(f"{city}_s1_snap_vv_vh_vvdiff_10m_aligned.tif"))

    if not candidates:
        candidates = sorted((root / city).glob(f"{city}_s1*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No S1 raster found for {city} under {root}")

    return candidates[0]


def find_label_raster(instance_root: Path, city: str) -> Path:
    root = instance_root / "labels"
    candidates = sorted((root / city).glob(f"{city}_label_final.tif"))

    if not candidates:
        candidates = sorted((root / city).glob(f"{city}_label*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No label raster found for {city} under {root}")

    return candidates[0]


def classify_border_connected_nodata(
    nodata_mask: np.ndarray,
    border_margin: int,
) -> np.ndarray:
    nodata_mask = nodata_mask.astype(bool, copy=False)
    height, width = nodata_mask.shape

    if not nodata_mask.any():
        return np.zeros_like(nodata_mask, dtype=bool)

    margin = max(1, int(border_margin))
    margin_rows = min(margin, height)
    margin_cols = min(margin, width)

    seed = np.zeros_like(nodata_mask, dtype=bool)
    seed[:margin_rows, :] = True
    seed[-margin_rows:, :] = True
    seed[:, :margin_cols] = True
    seed[:, -margin_cols:] = True
    seed &= nodata_mask

    if not seed.any():
        return np.zeros_like(nodata_mask, dtype=bool)

    try:
        from scipy import ndimage

        return ndimage.binary_propagation(seed, mask=nodata_mask).astype(bool)

    except Exception as exc:
        print(
            "[WARN] scipy.ndimage failed or unavailable. Falling back to BFS. "
            f"Reason: {exc}",
            file=sys.stderr,
        )

    visited = np.zeros_like(nodata_mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    rows, cols = np.where(seed)

    for r, c in zip(rows, cols):
        visited[int(r), int(c)] = True
        queue.append((int(r), int(c)))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()

        for dr, dc in neighbors:
            rr = r + dr
            cc = c + dc

            if rr < 0 or rr >= height or cc < 0 or cc >= width:
                continue

            if visited[rr, cc]:
                continue

            if not nodata_mask[rr, cc]:
                continue

            visited[rr, cc] = True
            queue.append((rr, cc))

    return visited


def build_multiband_nodata_mask(
    raster_path: Path,
    all_zero_as_nodata: bool,
    nan_as_nodata: bool,
) -> tuple[np.ndarray, dict]:
    """
    Build combined nodata mask for a multiband raster.

    combined nodata =
        official raster mask nodata in any band
        OR all-zero-all-band pixels, if requested
        OR non-finite values in any band, if requested
    """
    with rasterio.open(raster_path) as src:
        height = src.height
        width = src.width
        count = src.count

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
            block_combined = official_block

            official[row0:row1, col0:col1] = official_block

            if all_zero_as_nodata or nan_as_nodata:
                data = src.read(indexes=band_indexes, window=window)

                if all_zero_as_nodata:
                    all_zero_block = np.all(data == 0, axis=0)
                    all_zero[row0:row1, col0:col1] = all_zero_block
                    block_combined = block_combined | all_zero_block

                if nan_as_nodata:
                    nonfinite_block = np.any(~np.isfinite(data), axis=0)
                    nonfinite[row0:row1, col0:col1] = nonfinite_block
                    block_combined = block_combined | nonfinite_block

            combined[row0:row1, col0:col1] = block_combined

        meta = {
            "width": width,
            "height": height,
            "band_count": count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "nodata_value": src.nodata,
            "total_pixels": int(height * width),
            "official_nodata_pixels": int(official.sum()),
            "all_zero_allbands_pixels": int(all_zero.sum()),
            "nonfinite_pixels": int(nonfinite.sum()),
            "combined_nodata_pixels": int(combined.sum()),
        }

    return combined, meta


def inspect_nodata_components(
    nodata_mask: np.ndarray,
    border_margin: int,
) -> dict:
    total_pixels = int(nodata_mask.size)
    combined_pixels = int(nodata_mask.sum())

    border_connected = classify_border_connected_nodata(
        nodata_mask=nodata_mask,
        border_margin=border_margin,
    )

    internal = nodata_mask & ~border_connected

    border_pixels = int(border_connected.sum())
    internal_pixels = int(internal.sum())

    return {
        "combined_nodata_pixels": combined_pixels,
        "border_connected_nodata_pixels": border_pixels,
        "internal_nodata_pixels": internal_pixels,
        "combined_nodata_percent": percent(combined_pixels, total_pixels),
        "border_connected_nodata_percent": percent(border_pixels, total_pixels),
        "internal_nodata_percent": percent(internal_pixels, total_pixels),
        "border_share_percent": percent(border_pixels, combined_pixels),
        "internal_share_percent": percent(internal_pixels, combined_pixels),
    }


def inspect_label(label_path: Path, positive_threshold: float) -> dict:
    unique_values: set[float] = set()
    unique_overflow = False
    positive_pixels = 0
    total_pixels = 0
    label_min = None
    label_max = None
    mask_nodata_pixels = 0
    nonfinite_pixels = 0

    with rasterio.open(label_path) as src:
        for _, window in src.block_windows(1):
            data = src.read(1, window=window, masked=False)
            mask = src.read_masks(1, window=window)

            total_pixels += data.size
            positive_pixels += int(np.count_nonzero(data > positive_threshold))
            mask_nodata_pixels += int(np.count_nonzero(mask == 0))
            nonfinite_pixels += int(np.count_nonzero(~np.isfinite(data)))

            block_min = float(np.nanmin(data))
            block_max = float(np.nanmax(data))

            if label_min is None or block_min < label_min:
                label_min = block_min
            if label_max is None or block_max > label_max:
                label_max = block_max

            if not unique_overflow:
                vals = np.unique(data)
                for value in vals:
                    unique_values.add(float(value))
                    if len(unique_values) > 20:
                        unique_overflow = True
                        unique_values.clear()
                        break

        if unique_overflow:
            unique_repr = "more_than_20_unique_values"
            binary_ok = False
        else:
            sorted_values = sorted(unique_values)
            unique_repr = ",".join(
                str(int(v)) if float(v).is_integer() else str(v)
                for v in sorted_values
            )
            binary_ok = all(v in {0.0, 1.0} for v in sorted_values)

        return {
            "label_min": label_min,
            "label_max": label_max,
            "label_unique_values_limited": unique_repr,
            "label_binary_ok": binary_ok,
            "label_positive_pixels": positive_pixels,
            "label_positive_percent": percent(positive_pixels, total_pixels),
            "label_mask_nodata_pixels": mask_nodata_pixels,
            "label_mask_nodata_percent": percent(mask_nodata_pixels, total_pixels),
            "label_nonfinite_pixels": nonfinite_pixels,
            "label_nonfinite_percent": percent(nonfinite_pixels, total_pixels),
        }


def check_alignment(s2_path: Path, s1_path: Path, label_path: Path) -> dict:
    with rasterio.open(s2_path) as s2, rasterio.open(s1_path) as s1, rasterio.open(label_path) as lab:
        s1_same_width = s2.width == s1.width
        s1_same_height = s2.height == s1.height
        s1_same_crs = s2.crs == s1.crs
        s1_same_transform = s2.transform.almost_equals(s1.transform)

        label_same_width = s2.width == lab.width
        label_same_height = s2.height == lab.height
        label_same_crs = s2.crs == lab.crs
        label_same_transform = s2.transform.almost_equals(lab.transform)

        s1_alignment_ok = (
            s1_same_width
            and s1_same_height
            and s1_same_crs
            and s1_same_transform
        )

        label_alignment_ok = (
            label_same_width
            and label_same_height
            and label_same_crs
            and label_same_transform
        )

        return {
            "s2_width": s2.width,
            "s2_height": s2.height,
            "s2_band_count": s2.count,
            "s2_dtype": s2.dtypes[0],
            "s2_crs": str(s2.crs),
            "s2_transform": tuple(s2.transform),
            "s2_nodata_value": s2.nodata,
            "s1_width": s1.width,
            "s1_height": s1.height,
            "s1_band_count": s1.count,
            "s1_dtype": s1.dtypes[0],
            "s1_crs": str(s1.crs),
            "s1_transform": tuple(s1.transform),
            "s1_nodata_value": s1.nodata,
            "label_width": lab.width,
            "label_height": lab.height,
            "label_band_count": lab.count,
            "label_dtype": lab.dtypes[0],
            "label_crs": str(lab.crs),
            "label_transform": tuple(lab.transform),
            "label_nodata_value": lab.nodata,
            "s1_same_width": s1_same_width,
            "s1_same_height": s1_same_height,
            "s1_same_crs": s1_same_crs,
            "s1_same_transform": s1_same_transform,
            "s1_alignment_ok": s1_alignment_ok,
            "label_same_width": label_same_width,
            "label_same_height": label_same_height,
            "label_same_crs": label_same_crs,
            "label_same_transform": label_same_transform,
            "label_alignment_ok": label_alignment_ok,
            "all_alignment_ok": s1_alignment_ok and label_alignment_ok,
        }


def classify_city_status(
    row: dict,
    expected_s2_bands: int,
    expected_s1_bands: int,
    expected_label_bands: int,
) -> tuple[str, str]:
    if not row["all_alignment_ok"]:
        return "fail_alignment", "S2/S1/label grids are not aligned"

    if row["s2_band_count"] != expected_s2_bands:
        return "fail_s2_band_count", f"S2 band count is {row['s2_band_count']}"

    if row["s1_band_count"] != expected_s1_bands:
        return "fail_s1_band_count", f"S1 band count is {row['s1_band_count']}"

    if row["label_band_count"] != expected_label_bands:
        return "fail_label_band_count", f"label band count is {row['label_band_count']}"

    if not row["label_binary_ok"]:
        return "fail_label_not_binary", "label raster contains values outside 0/1"

    if row["s2_combined_nodata_pixels"] > 0:
        return (
            "pass_alignment_residual_s2_nodata",
            "alignment passes, but residual S2 nodata remains after crop",
        )

    if row["s1_combined_nodata_pixels"] > 0:
        return (
            "pass_alignment_residual_s1_nodata",
            "alignment passes, S2 clean, but residual S1 nodata remains",
        )

    return "pass_clean", "alignment passes and S2/S1 contain no detected nodata"


def inspect_city(city: str, instance_root: Path, args: argparse.Namespace) -> dict:
    s2_path = find_s2_raster(instance_root, city)
    s1_path = find_s1_raster(instance_root, city)
    label_path = find_label_raster(instance_root, city)

    alignment = check_alignment(s2_path, s1_path, label_path)

    s2_mask, s2_meta = build_multiband_nodata_mask(
        raster_path=s2_path,
        all_zero_as_nodata=args.s2_all_zero_as_nodata,
        nan_as_nodata=args.nan_as_nodata,
    )
    s2_nodata = inspect_nodata_components(
        nodata_mask=s2_mask,
        border_margin=args.border_margin,
    )

    s1_mask, s1_meta = build_multiband_nodata_mask(
        raster_path=s1_path,
        all_zero_as_nodata=args.s1_all_zero_as_nodata,
        nan_as_nodata=args.nan_as_nodata,
    )
    s1_nodata = inspect_nodata_components(
        nodata_mask=s1_mask,
        border_margin=args.border_margin,
    )

    label_stats = inspect_label(
        label_path=label_path,
        positive_threshold=args.positive_threshold,
    )

    row = {
        "city": city,
        "s2_path": str(s2_path),
        "s1_path": str(s1_path),
        "label_path": str(label_path),
        **alignment,
        "s2_official_nodata_pixels": s2_meta["official_nodata_pixels"],
        "s2_all_zero_allbands_pixels": s2_meta["all_zero_allbands_pixels"],
        "s2_nonfinite_pixels": s2_meta["nonfinite_pixels"],
        "s2_combined_nodata_pixels": s2_nodata["combined_nodata_pixels"],
        "s2_border_connected_nodata_pixels": s2_nodata["border_connected_nodata_pixels"],
        "s2_internal_nodata_pixels": s2_nodata["internal_nodata_pixels"],
        "s2_combined_nodata_percent": s2_nodata["combined_nodata_percent"],
        "s2_border_connected_nodata_percent": s2_nodata["border_connected_nodata_percent"],
        "s2_internal_nodata_percent": s2_nodata["internal_nodata_percent"],
        "s2_border_share_percent": s2_nodata["border_share_percent"],
        "s2_internal_share_percent": s2_nodata["internal_share_percent"],
        "s1_official_nodata_pixels": s1_meta["official_nodata_pixels"],
        "s1_all_zero_allbands_pixels": s1_meta["all_zero_allbands_pixels"],
        "s1_nonfinite_pixels": s1_meta["nonfinite_pixels"],
        "s1_combined_nodata_pixels": s1_nodata["combined_nodata_pixels"],
        "s1_border_connected_nodata_pixels": s1_nodata["border_connected_nodata_pixels"],
        "s1_internal_nodata_pixels": s1_nodata["internal_nodata_pixels"],
        "s1_combined_nodata_percent": s1_nodata["combined_nodata_percent"],
        "s1_border_connected_nodata_percent": s1_nodata["border_connected_nodata_percent"],
        "s1_internal_nodata_percent": s1_nodata["internal_nodata_percent"],
        "s1_border_share_percent": s1_nodata["border_share_percent"],
        "s1_internal_share_percent": s1_nodata["internal_share_percent"],
        **label_stats,
        "border_margin_pixels": args.border_margin,
        "s2_all_zero_as_nodata": args.s2_all_zero_as_nodata,
        "s1_all_zero_as_nodata": args.s1_all_zero_as_nodata,
        "nan_as_nodata": args.nan_as_nodata,
    }

    city_status, city_status_reason = classify_city_status(
        row=row,
        expected_s2_bands=args.expected_s2_bands,
        expected_s1_bands=args.expected_s1_bands,
        expected_label_bands=args.expected_label_bands,
    )

    row["city_status"] = city_status
    row["city_status_reason"] = city_status_reason

    return row


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
        "city_status",
        "all_alignment_ok",
        "s2_band_count",
        "s1_band_count",
        "label_band_count",
        "label_binary_ok",
        "s2_combined_nodata_percent",
        "s2_border_connected_nodata_percent",
        "s2_internal_nodata_percent",
        "s1_combined_nodata_percent",
        "label_positive_pixels",
        "label_positive_percent",
        "city_status_reason",
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Instance C nodata and alignment QC\n\n")
        f.write(
            "This report validates the cropped repaired dataset instance. "
            "It checks raster existence, band counts, grid alignment, label binary values, "
            "and residual nodata.\n\n"
        )

        f.write("## Status counts\n\n")
        counts: dict[str, int] = {}

        for row in rows:
            status = row["city_status"]
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

    s2_root = instance_root / "s2"
    s1_root = instance_root / "s1_snap"
    labels_root = instance_root / "labels"

    for path, name in [
        (s2_root, "S2 root"),
        (s1_root, "S1 root"),
        (labels_root, "labels root"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    output_dir = Path(args.output_dir) if args.output_dir else instance_root / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    cities = discover_cities(s2_root, args.cities)

    print(f"[INFO] Instance root: {instance_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Cities to inspect: {len(cities)}")
    print(f"[INFO] S2 all-zero-as-nodata: {args.s2_all_zero_as_nodata}")
    print(f"[INFO] S1 all-zero-as-nodata: {args.s1_all_zero_as_nodata}")
    print(f"[INFO] nan-as-nodata: {args.nan_as_nodata}")
    print(f"[INFO] border_margin: {args.border_margin}")

    rows: list[dict] = []

    for i, city in enumerate(cities, start=1):
        print(f"\n[STEP {i}/{len(cities)}] {city}")

        try:
            row = inspect_city(city=city, instance_root=instance_root, args=args)
            rows.append(row)

            print(
                "[OK] "
                f"status={row['city_status']} | "
                f"alignment={row['all_alignment_ok']} | "
                f"S2 nodata={row['s2_combined_nodata_percent']:.6f}% | "
                f"S2 internal={row['s2_internal_nodata_percent']:.6f}% | "
                f"S1 nodata={row['s1_combined_nodata_percent']:.6f}% | "
                f"label_binary={row['label_binary_ok']}"
            )

        except Exception as exc:
            print(f"[ERROR] {city}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "city": city,
                    "city_status": "error",
                    "city_status_reason": str(exc),
                    "all_alignment_ok": False,
                    "s2_combined_nodata_percent": "",
                    "s1_combined_nodata_percent": "",
                    "label_binary_ok": "",
                }
            )

    rows = sorted(rows, key=lambda row: str(row["city"]))

    csv_path = output_dir / "instance_C_nodata_alignment_summary.csv"
    json_path = output_dir / "instance_C_nodata_alignment_summary.json"
    md_path = output_dir / "instance_C_nodata_alignment_summary.md"

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
        status = row["city_status"]
        counts[status] = counts.get(status, 0) + 1

    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()