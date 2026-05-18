#!/usr/bin/env python3
"""
Plan residual Sentinel-1 nodata handling after S2 repair.

This script does not modify rasters.

It inspects S1 nodata in the repaired instance and produces a per-city decision
table describing whether S1 nodata can be ignored, filled, reviewed, or must be
handled by reprocessing/regeneration/patch filtering.

Context:
- S2 has already been repaired and validated in s2_filled/.
- Remaining nodata is now an S1-specific issue.
- Large S1 footprint gaps must not be filled blindly.
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
        description="Plan residual S1 nodata handling in repaired instance C."
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
        "--s1-subdir",
        type=str,
        default="s1_snap",
        help="S1 subdirectory inside instance root. Default: s1_snap",
    )

    parser.add_argument(
        "--labels-subdir",
        type=str,
        default="labels",
        help="Labels subdirectory inside instance root. Default: labels",
    )

    parser.add_argument(
        "--s2-subdir",
        type=str,
        default="s2_filled",
        help="Optional S2-filled subdirectory for alignment checking. Default: s2_filled",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. If omitted, outputs are written to "
            "<instance-root>/qc/s1_nodata_plan"
        ),
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional city list. If omitted, all cities under S1 root are processed.",
    )

    s1_zero_group = parser.add_mutually_exclusive_group()
    s1_zero_group.add_argument(
        "--s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_true",
        help="Treat pixels where all S1 bands are exactly zero as nodata.",
    )
    s1_zero_group.add_argument(
        "--no-s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_false",
        help="Do not treat all-zero S1 pixels as nodata.",
    )
    parser.set_defaults(s1_all_zero_as_nodata=False)

    nan_group = parser.add_mutually_exclusive_group()
    nan_group.add_argument(
        "--nan-as-nodata",
        dest="nan_as_nodata",
        action="store_true",
        help="Treat NaN/Inf values in S1 as nodata.",
    )
    nan_group.add_argument(
        "--no-nan-as-nodata",
        dest="nan_as_nodata",
        action="store_false",
        help="Do not treat NaN/Inf values as nodata.",
    )
    parser.set_defaults(nan_as_nodata=True)

    parser.add_argument(
        "--border-margin",
        type=int,
        default=128,
        help="Border margin in pixels for border-connected nodata classification.",
    )

    parser.add_argument(
        "--label-positive-threshold",
        type=float,
        default=0.5,
        help="Label pixels > this threshold are considered positive. Default: 0.5",
    )

    parser.add_argument(
        "--tiny-s1-nodata-threshold-percent",
        type=float,
        default=0.5,
        help=(
            "S1 nodata below or equal to this percentage is considered tiny. "
            "Default: 0.5%%"
        ),
    )

    parser.add_argument(
        "--large-s1-nodata-threshold-percent",
        type=float,
        default=5.0,
        help=(
            "S1 nodata above or equal to this percentage is considered a major "
            "coverage/footprint issue. Default: 5.0%%"
        ),
    )

    parser.add_argument(
        "--tiny-label-overlap-threshold-percent",
        type=float,
        default=0.1,
        help=(
            "If S1 nodata overlaps this percentage or less of positive label pixels, "
            "the overlap is considered tiny. Default: 0.1%%"
        ),
    )

    parser.add_argument(
        "--significant-label-overlap-threshold-percent",
        type=float,
        default=1.0,
        help=(
            "If S1 nodata overlaps this percentage or more of positive label pixels, "
            "the overlap is considered significant. Default: 1.0%%"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing plan outputs.",
    )

    return parser.parse_args()


def percent(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def discover_cities(root: Path, cities: list[str] | None) -> list[str]:
    if cities:
        return sorted(cities)

    discovered = sorted([p.name for p in root.iterdir() if p.is_dir()])

    if not discovered:
        raise FileNotFoundError(f"No city folders found under {root}")

    return discovered


def find_s1_raster(instance_root: Path, s1_subdir: str, city: str) -> Path:
    city_dir = instance_root / s1_subdir / city

    candidates = sorted(city_dir.glob(f"{city}_s1_snap_vv_vh_vvdiff_10m_aligned.tif"))

    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s1*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No S1 raster found for {city} under {city_dir}")

    return candidates[0]


def find_label_raster(instance_root: Path, labels_subdir: str, city: str) -> Path:
    city_dir = instance_root / labels_subdir / city

    candidates = sorted(city_dir.glob(f"{city}_label_final.tif"))

    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_label*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No label raster found for {city} under {city_dir}")

    return candidates[0]


def find_s2_raster(instance_root: Path, s2_subdir: str, city: str) -> Path | None:
    city_dir = instance_root / s2_subdir / city

    if not city_dir.exists():
        return None

    candidates = sorted(city_dir.glob(f"{city}_s2_12bands_reflectance_10m.tif"))

    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s2*.tif"))

    if not candidates:
        return None

    return candidates[0]


def classify_border_connected_nodata(
    nodata_mask: np.ndarray,
    border_margin: int,
) -> np.ndarray:
    nodata_mask = nodata_mask.astype(bool, copy=False)

    if not nodata_mask.any():
        return np.zeros_like(nodata_mask, dtype=bool)

    height, width = nodata_mask.shape
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
            "[WARN] scipy.ndimage unavailable or failed; using BFS. "
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


def build_s1_nodata_mask(
    s1_path: Path,
    all_zero_as_nodata: bool,
    nan_as_nodata: bool,
) -> tuple[np.ndarray, dict]:
    """
    Build full-resolution S1 nodata mask.

    combined nodata =
        official raster mask nodata in any band
        OR all-zero-all-band pixels, if requested
        OR non-finite values in any band, if requested
    """
    with rasterio.open(s1_path) as src:
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
            "nodata_value": src.nodata,
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


def inspect_label_overlap(
    label_path: Path,
    s1_nodata_mask: np.ndarray,
    positive_threshold: float,
) -> dict:
    positive_pixels = 0
    positive_on_s1_nodata = 0
    total_pixels = 0
    label_min = None
    label_max = None
    unique_values: set[float] = set()
    unique_overflow = False

    with rasterio.open(label_path) as src:
        if src.height != s1_nodata_mask.shape[0] or src.width != s1_nodata_mask.shape[1]:
            raise ValueError(
                f"Label shape does not match S1 nodata mask. "
                f"Label={src.width}x{src.height}; "
                f"S1 mask={s1_nodata_mask.shape[1]}x{s1_nodata_mask.shape[0]}"
            )

        for _, window in src.block_windows(1):
            row0 = int(window.row_off)
            row1 = int(window.row_off + window.height)
            col0 = int(window.col_off)
            col1 = int(window.col_off + window.width)

            data = src.read(1, window=window, masked=False)
            positive = data > positive_threshold
            nodata_block = s1_nodata_mask[row0:row1, col0:col1]

            total_pixels += int(data.size)
            positive_pixels += int(np.count_nonzero(positive))
            positive_on_s1_nodata += int(np.count_nonzero(positive & nodata_block))

            block_min = float(np.nanmin(data))
            block_max = float(np.nanmax(data))

            if label_min is None or block_min < label_min:
                label_min = block_min

            if label_max is None or block_max > label_max:
                label_max = block_max

            if not unique_overflow:
                values = np.unique(data)
                for value in values:
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
        "label_total_pixels": total_pixels,
        "label_positive_pixels": positive_pixels,
        "label_positive_percent": percent(positive_pixels, total_pixels),
        "label_positive_on_s1_nodata_pixels": positive_on_s1_nodata,
        "label_positive_on_s1_nodata_percent_of_label": percent(
            positive_on_s1_nodata, positive_pixels
        ),
        "label_positive_on_s1_nodata_percent_of_raster": percent(
            positive_on_s1_nodata, total_pixels
        ),
        "label_min": label_min,
        "label_max": label_max,
        "label_unique_values_limited": unique_repr,
        "label_binary_ok": binary_ok,
    }


def check_alignment(reference_path: Path, other_path: Path, other_name: str) -> dict:
    with rasterio.open(reference_path) as ref, rasterio.open(other_path) as other:
        same_width = ref.width == other.width
        same_height = ref.height == other.height
        same_crs = ref.crs == other.crs
        same_transform = ref.transform.almost_equals(other.transform)

        return {
            f"{other_name}_same_width": same_width,
            f"{other_name}_same_height": same_height,
            f"{other_name}_same_crs": same_crs,
            f"{other_name}_same_transform": same_transform,
            f"{other_name}_alignment_ok": (
                same_width and same_height and same_crs and same_transform
            ),
        }


def recommend_action(
    s1_nodata_percent: float,
    s1_internal_percent: float,
    label_overlap_percent_of_label: float,
    args: argparse.Namespace,
) -> tuple[str, str]:
    if s1_nodata_percent == 0:
        return "no_action", "S1 contains no detected nodata."

    if (
        s1_nodata_percent >= args.large_s1_nodata_threshold_percent
        or label_overlap_percent_of_label >= args.significant_label_overlap_threshold_percent
    ):
        return (
            "major_s1_coverage_issue_do_not_fill",
            (
                "S1 nodata is large and/or overlaps a significant share of labels. "
                "Do not use spatial interpolation. Prefer S1 reprocessing/regeneration, "
                "additional S1 product selection, stricter joint valid mask, or S1-aware patch filtering."
            ),
        )

    if s1_nodata_percent <= args.tiny_s1_nodata_threshold_percent:
        if label_overlap_percent_of_label <= args.tiny_label_overlap_threshold_percent:
            return (
                "fill_tiny_s1_nodata_candidate",
                (
                    "S1 nodata is tiny and has negligible positive-label overlap. "
                    "It can likely be filled conservatively or handled by patch filtering."
                ),
            )

        return (
            "review_tiny_s1_nodata_label_overlap",
            (
                "S1 nodata is tiny but overlaps labels above the tiny-overlap threshold. "
                "Review visually before filling or filtering."
            ),
        )

    return (
        "moderate_s1_nodata_review",
        (
            "S1 nodata is not huge, but it is above the tiny threshold. "
            "Review footprint/label overlap before deciding between fill, crop, regeneration, or patch filtering."
        ),
    )


def inspect_city(city: str, instance_root: Path, args: argparse.Namespace) -> dict:
    s1_path = find_s1_raster(instance_root, args.s1_subdir, city)
    label_path = find_label_raster(instance_root, args.labels_subdir, city)
    s2_path = find_s2_raster(instance_root, args.s2_subdir, city)

    s1_mask, s1_meta = build_s1_nodata_mask(
        s1_path=s1_path,
        all_zero_as_nodata=args.s1_all_zero_as_nodata,
        nan_as_nodata=args.nan_as_nodata,
    )

    border_mask = classify_border_connected_nodata(
        nodata_mask=s1_mask,
        border_margin=args.border_margin,
    )
    internal_mask = s1_mask & ~border_mask

    s1_nodata_pixels = int(s1_mask.sum())
    s1_border_pixels = int(border_mask.sum())
    s1_internal_pixels = int(internal_mask.sum())
    total_pixels = int(s1_mask.size)

    label_stats = inspect_label_overlap(
        label_path=label_path,
        s1_nodata_mask=s1_mask,
        positive_threshold=args.label_positive_threshold,
    )

    label_alignment = check_alignment(s1_path, label_path, "label")

    if s2_path is not None:
        s2_alignment = check_alignment(s1_path, s2_path, "s2_filled")
        s2_path_str = str(s2_path)
    else:
        s2_alignment = {
            "s2_filled_same_width": "",
            "s2_filled_same_height": "",
            "s2_filled_same_crs": "",
            "s2_filled_same_transform": "",
            "s2_filled_alignment_ok": "",
        }
        s2_path_str = ""

    label_overlap_percent = label_stats["label_positive_on_s1_nodata_percent_of_label"]

    action, reason = recommend_action(
        s1_nodata_percent=percent(s1_nodata_pixels, total_pixels),
        s1_internal_percent=percent(s1_internal_pixels, total_pixels),
        label_overlap_percent_of_label=label_overlap_percent,
        args=args,
    )

    return {
        "city": city,
        "recommended_action": action,
        "reason": reason,
        "s1_path": str(s1_path),
        "label_path": str(label_path),
        "s2_filled_path": s2_path_str,
        "width": s1_meta["width"],
        "height": s1_meta["height"],
        "band_count": s1_meta["band_count"],
        "dtype": s1_meta["dtype"],
        "crs": s1_meta["crs"],
        "nodata_value": s1_meta["nodata_value"],
        "total_pixels": total_pixels,
        "s1_official_nodata_pixels": s1_meta["official_nodata_pixels"],
        "s1_all_zero_allbands_pixels": s1_meta["all_zero_allbands_pixels"],
        "s1_nonfinite_pixels": s1_meta["nonfinite_pixels"],
        "s1_combined_nodata_pixels": s1_nodata_pixels,
        "s1_border_connected_nodata_pixels": s1_border_pixels,
        "s1_internal_nodata_pixels": s1_internal_pixels,
        "s1_official_nodata_percent": s1_meta["official_nodata_percent"],
        "s1_all_zero_allbands_percent": s1_meta["all_zero_allbands_percent"],
        "s1_nonfinite_percent": s1_meta["nonfinite_percent"],
        "s1_combined_nodata_percent": percent(s1_nodata_pixels, total_pixels),
        "s1_border_connected_nodata_percent": percent(s1_border_pixels, total_pixels),
        "s1_internal_nodata_percent": percent(s1_internal_pixels, total_pixels),
        "s1_border_share_percent": percent(s1_border_pixels, s1_nodata_pixels),
        "s1_internal_share_percent": percent(s1_internal_pixels, s1_nodata_pixels),
        **label_stats,
        **label_alignment,
        **s2_alignment,
        "border_margin_pixels": args.border_margin,
        "s1_all_zero_as_nodata": args.s1_all_zero_as_nodata,
        "nan_as_nodata": args.nan_as_nodata,
        "tiny_s1_nodata_threshold_percent": args.tiny_s1_nodata_threshold_percent,
        "large_s1_nodata_threshold_percent": args.large_s1_nodata_threshold_percent,
        "tiny_label_overlap_threshold_percent": args.tiny_label_overlap_threshold_percent,
        "significant_label_overlap_threshold_percent": (
            args.significant_label_overlap_threshold_percent
        ),
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
        "recommended_action",
        "s1_combined_nodata_percent",
        "s1_border_connected_nodata_percent",
        "s1_internal_nodata_percent",
        "label_positive_on_s1_nodata_percent_of_label",
        "label_positive_on_s1_nodata_pixels",
        "label_positive_pixels",
        "reason",
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Residual S1 nodata handling plan\n\n")
        f.write(
            "This report plans how to handle residual S1 nodata after the S2 repair. "
            "It does not modify rasters.\n\n"
        )

        f.write("## Recommended action counts\n\n")
        counts: dict[str, int] = {}

        for row in rows:
            action = row["recommended_action"]
            counts[action] = counts.get(action, 0) + 1

        for action, count in sorted(counts.items()):
            f.write(f"- `{action}`: {count}\n")

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

    s1_root = instance_root / args.s1_subdir
    labels_root = instance_root / args.labels_subdir

    if not s1_root.exists():
        raise FileNotFoundError(f"S1 root does not exist: {s1_root}")

    if not labels_root.exists():
        raise FileNotFoundError(f"Labels root does not exist: {labels_root}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else instance_root / "qc" / "s1_nodata_plan"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cities = discover_cities(s1_root, args.cities)

    print(f"[INFO] Instance root: {instance_root}")
    print(f"[INFO] S1 root: {s1_root}")
    print(f"[INFO] Labels root: {labels_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Cities to inspect: {len(cities)}")
    print(f"[INFO] s1_all_zero_as_nodata: {args.s1_all_zero_as_nodata}")
    print(f"[INFO] nan_as_nodata: {args.nan_as_nodata}")
    print(f"[INFO] tiny_s1_nodata_threshold_percent: {args.tiny_s1_nodata_threshold_percent}")
    print(f"[INFO] large_s1_nodata_threshold_percent: {args.large_s1_nodata_threshold_percent}")

    rows: list[dict] = []

    for i, city in enumerate(cities, start=1):
        print(f"\n[STEP {i}/{len(cities)}] {city}")

        try:
            row = inspect_city(city=city, instance_root=instance_root, args=args)
            rows.append(row)

            print(
                "[OK] "
                f"action={row['recommended_action']} | "
                f"S1 nodata={row['s1_combined_nodata_percent']:.6f}% | "
                f"S1 internal={row['s1_internal_nodata_percent']:.6f}% | "
                f"label overlap={row['label_positive_on_s1_nodata_percent_of_label']:.6f}%"
            )

        except Exception as exc:
            print(f"[ERROR] {city}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "city": city,
                    "recommended_action": "error",
                    "reason": str(exc),
                }
            )

    rows = sorted(rows, key=lambda row: str(row["city"]))

    csv_path = output_dir / "s1_nodata_handling_plan.csv"
    json_path = output_dir / "s1_nodata_handling_plan.json"
    md_path = output_dir / "s1_nodata_handling_plan.md"

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
        action = row.get("recommended_action", "unknown")
        counts[action] = counts.get(action, 0) + 1

    for action, count in sorted(counts.items()):
        print(f"  {action}: {count}")


if __name__ == "__main__":
    main()