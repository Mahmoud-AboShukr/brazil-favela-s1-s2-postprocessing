#!/usr/bin/env python3
"""
Build a full-resolution Sentinel-2 nodata repair plan.

This script inspects city-level standardized Sentinel-2 rasters and classifies
nodata into:

1. combined nodata
2. border-connected nodata
3. internal nodata

It then assigns a recommended repair action per city:

- no_action
- crop_border
- crop_border_then_verify_tiny_internal
- crop_border_plus_internal_fill

Important:
This script does not modify any raster. It only creates CSV/Markdown reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full-resolution S2 nodata repair plan for all city rasters."
    )

    parser.add_argument(
        "--s2-root",
        type=str,
        required=True,
        help=(
            "Root folder containing city S2 rasters, e.g. "
            "D:/post_processing_dataset/dataset_instances/instance_B_standard_rs/s2"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where repair-plan CSV/JSON/Markdown outputs will be written.",
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help=(
            "Optional list of city slugs to inspect. "
            "If omitted, all cities under --s2-root are processed."
        ),
    )

    zero_group = parser.add_mutually_exclusive_group()
    zero_group.add_argument(
        "--all-zero-as-nodata",
        dest="all_zero_as_nodata",
        action="store_true",
        help="Treat pixels where all bands are exactly zero as nodata.",
    )
    zero_group.add_argument(
        "--no-all-zero-as-nodata",
        dest="all_zero_as_nodata",
        action="store_false",
        help="Do not treat all-zero-all-band pixels as nodata.",
    )
    parser.set_defaults(all_zero_as_nodata=True)

    nan_group = parser.add_mutually_exclusive_group()
    nan_group.add_argument(
        "--nan-as-nodata",
        dest="nan_as_nodata",
        action="store_true",
        help="Treat pixels with NaN/Inf in any band as nodata.",
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
        help=(
            "Border seed margin in original full-resolution pixels. "
            "Nodata connected to this border seed zone is classified as border-connected."
        ),
    )

    parser.add_argument(
        "--no-action-total-threshold-percent",
        type=float,
        default=0.01,
        help=(
            "If total combined nodata percentage is at or below this value, "
            "recommended_action becomes no_action."
        ),
    )

    parser.add_argument(
        "--internal-fill-threshold-percent",
        type=float,
        default=0.10,
        help=(
            "If internal nodata percentage of total raster pixels is at or above this value, "
            "recommended_action becomes crop_border_plus_internal_fill."
        ),
    )

    parser.add_argument(
        "--mostly-border-share-percent",
        type=float,
        default=95.0,
        help=(
            "If border-connected nodata represents at least this share of all nodata, "
            "and internal nodata is below the fill threshold, action becomes crop_border."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    return parser.parse_args()


def find_city_raster(s2_root: Path, city: str) -> Path:
    city_dir = s2_root / city

    patterns = [
        f"{city}_s2_12bands_reflectance_10m.tif",
        f"{city}_s2*.tif",
        "*_s2_12bands_reflectance_10m.tif",
        "*.tif",
    ]

    candidates: list[Path] = []

    if city_dir.exists():
        for pattern in patterns:
            candidates.extend(sorted(city_dir.glob(pattern)))

    if not candidates:
        for pattern in patterns:
            candidates.extend(sorted(s2_root.glob(pattern)))
            candidates.extend(sorted(s2_root.glob(f"*/{pattern}")))

    candidates = [p for p in candidates if p.is_file() and city in p.name]

    if not candidates:
        raise FileNotFoundError(f"No S2 raster found for city '{city}' under {s2_root}")

    exact = [p for p in candidates if p.name == f"{city}_s2_12bands_reflectance_10m.tif"]
    if exact:
        return exact[0]

    return candidates[0]


def discover_city_rasters(s2_root: Path, cities: Iterable[str] | None) -> dict[str, Path]:
    if cities:
        return {city: find_city_raster(s2_root, city) for city in cities}

    discovered: dict[str, Path] = {}

    for path in sorted(s2_root.glob("*/*_s2_12bands_reflectance_10m.tif")):
        discovered[path.parent.name] = path

    if not discovered:
        for path in sorted(s2_root.glob("*_s2_12bands_reflectance_10m.tif")):
            city = path.name.replace("_s2_12bands_reflectance_10m.tif", "")
            discovered[city] = path

    if not discovered:
        raise FileNotFoundError(f"No S2 rasters discovered under {s2_root}")

    return discovered


def percent(count: int | float, total: int | float) -> float:
    if total == 0:
        return 0.0
    return 100.0 * float(count) / float(total)


def build_combined_nodata_mask(
    raster_path: Path,
    all_zero_as_nodata: bool,
    nan_as_nodata: bool,
) -> tuple[np.ndarray, dict]:
    """
    Build a full-resolution combined nodata mask.

    The combined mask is:

        official raster mask nodata
        OR all-zero-all-band pixels, if requested
        OR non-finite pixels, if requested

    The function reads by raster blocks to avoid loading the full 12-band raster
    into memory at once.
    """
    with rasterio.open(raster_path) as src:
        height = src.height
        width = src.width
        count = src.count

        combined = np.zeros((height, width), dtype=bool)
        official_nodata = np.zeros((height, width), dtype=bool)
        all_zero_nodata = np.zeros((height, width), dtype=bool)
        nonfinite_nodata = np.zeros((height, width), dtype=bool)

        band_indexes = list(range(1, count + 1))

        for _, window in src.block_windows(1):
            row0 = int(window.row_off)
            row1 = int(window.row_off + window.height)
            col0 = int(window.col_off)
            col1 = int(window.col_off + window.width)

            masks = src.read_masks(indexes=band_indexes, window=window)
            official_block = np.any(masks == 0, axis=0)
            official_nodata[row0:row1, col0:col1] = official_block

            block_combined = official_block

            if all_zero_as_nodata or nan_as_nodata:
                data = src.read(indexes=band_indexes, window=window)

                if all_zero_as_nodata:
                    all_zero_block = np.all(data == 0, axis=0)
                    all_zero_nodata[row0:row1, col0:col1] = all_zero_block
                    block_combined = block_combined | all_zero_block

                if nan_as_nodata:
                    nonfinite_block = np.any(~np.isfinite(data), axis=0)
                    nonfinite_nodata[row0:row1, col0:col1] = nonfinite_block
                    block_combined = block_combined | nonfinite_block

            combined[row0:row1, col0:col1] = block_combined

        metadata = {
            "width": width,
            "height": height,
            "band_count": count,
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "dtype": src.dtypes[0],
            "raster_nodata_value": src.nodata,
            "total_pixels": int(height * width),
            "official_nodata_pixels": int(official_nodata.sum()),
            "all_zero_allbands_pixels": int(all_zero_nodata.sum()),
            "nonfinite_pixels": int(nonfinite_nodata.sum()),
            "combined_nodata_pixels": int(combined.sum()),
        }

    return combined, metadata


def classify_border_connected_nodata(
    nodata_mask: np.ndarray,
    border_margin: int,
) -> np.ndarray:
    """
    Classify nodata pixels connected to the border seed zone.

    The seed zone is a margin around the full raster edge. Any nodata component
    touching this seed zone is considered border-connected.
    """
    nodata_mask = nodata_mask.astype(bool, copy=False)
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

    try:
        from scipy import ndimage

        return ndimage.binary_propagation(seed, mask=nodata_mask).astype(bool)

    except Exception as exc:
        print(
            "[WARN] scipy.ndimage is not available or failed. "
            "Falling back to pure-Python BFS, which may be slow. "
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


def recommend_action(
    combined_percent: float,
    border_share_percent: float,
    internal_percent: float,
    no_action_total_threshold_percent: float,
    internal_fill_threshold_percent: float,
    mostly_border_share_percent: float,
) -> tuple[str, str]:
    if combined_percent <= no_action_total_threshold_percent:
        return (
            "no_action",
            (
                f"combined nodata <= {no_action_total_threshold_percent:.3f}% "
                "so no repair is needed"
            ),
        )

    if internal_percent >= internal_fill_threshold_percent:
        return (
            "crop_border_plus_internal_fill",
            (
                f"internal nodata >= {internal_fill_threshold_percent:.3f}% "
                "of raster pixels; crop border first, then fill remaining internal nodata"
            ),
        )

    if border_share_percent >= mostly_border_share_percent:
        return (
            "crop_border",
            (
                f"border-connected nodata share >= {mostly_border_share_percent:.1f}% "
                "and internal nodata is below fill threshold"
            ),
        )

    return (
        "crop_border_then_verify_tiny_internal",
        (
            "internal nodata is below fill threshold but nodata is not overwhelmingly "
            "border-connected; crop border first, then verify residual internal pixels"
        ),
    )


def inspect_city(
    city: str,
    raster_path: Path,
    args: argparse.Namespace,
) -> dict:
    nodata_mask, meta = build_combined_nodata_mask(
        raster_path=raster_path,
        all_zero_as_nodata=args.all_zero_as_nodata,
        nan_as_nodata=args.nan_as_nodata,
    )

    border_connected = classify_border_connected_nodata(
        nodata_mask=nodata_mask,
        border_margin=args.border_margin,
    )

    internal = nodata_mask & ~border_connected

    total_pixels = meta["total_pixels"]
    combined_pixels = int(nodata_mask.sum())
    border_pixels = int(border_connected.sum())
    internal_pixels = int(internal.sum())

    combined_percent = percent(combined_pixels, total_pixels)
    official_percent = percent(meta["official_nodata_pixels"], total_pixels)
    all_zero_percent = percent(meta["all_zero_allbands_pixels"], total_pixels)
    nonfinite_percent = percent(meta["nonfinite_pixels"], total_pixels)
    border_percent = percent(border_pixels, total_pixels)
    internal_percent = percent(internal_pixels, total_pixels)

    border_share_percent = percent(border_pixels, combined_pixels)
    internal_share_percent = percent(internal_pixels, combined_pixels)

    action, reason = recommend_action(
        combined_percent=combined_percent,
        border_share_percent=border_share_percent,
        internal_percent=internal_percent,
        no_action_total_threshold_percent=args.no_action_total_threshold_percent,
        internal_fill_threshold_percent=args.internal_fill_threshold_percent,
        mostly_border_share_percent=args.mostly_border_share_percent,
    )

    return {
        "city": city,
        "recommended_action": action,
        "reason": reason,
        "s2_path": str(raster_path),
        "width": meta["width"],
        "height": meta["height"],
        "band_count": meta["band_count"],
        "dtype": meta["dtype"],
        "crs": meta["crs"],
        "raster_nodata_value": meta["raster_nodata_value"],
        "total_pixels": total_pixels,
        "official_nodata_pixels": meta["official_nodata_pixels"],
        "all_zero_allbands_pixels": meta["all_zero_allbands_pixels"],
        "nonfinite_pixels": meta["nonfinite_pixels"],
        "combined_nodata_pixels": combined_pixels,
        "border_connected_nodata_pixels": border_pixels,
        "internal_nodata_pixels": internal_pixels,
        "official_nodata_percent": official_percent,
        "all_zero_allbands_percent": all_zero_percent,
        "nonfinite_percent": nonfinite_percent,
        "combined_nodata_percent": combined_percent,
        "border_connected_nodata_percent": border_percent,
        "internal_nodata_percent": internal_percent,
        "border_share_percent": border_share_percent,
        "internal_share_percent": internal_share_percent,
        "border_margin_pixels": args.border_margin,
        "all_zero_as_nodata": args.all_zero_as_nodata,
        "nan_as_nodata": args.nan_as_nodata,
    }


def write_csv(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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

    selected_cols = [
        "city",
        "recommended_action",
        "combined_nodata_percent",
        "border_share_percent",
        "internal_share_percent",
        "internal_nodata_percent",
        "reason",
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Sentinel-2 nodata repair plan\n\n")
        f.write(
            "This report was generated from full-resolution city-level S2 rasters. "
            "It classifies nodata into border-connected and internal components.\n\n"
        )

        f.write("## Summary by recommended action\n\n")
        action_counts: dict[str, int] = {}
        for row in rows:
            action = row["recommended_action"]
            action_counts[action] = action_counts.get(action, 0) + 1

        for action, count in sorted(action_counts.items()):
            f.write(f"- `{action}`: {count} cities\n")

        f.write("\n## City table\n\n")
        f.write("| " + " | ".join(selected_cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(selected_cols)) + " |\n")

        for row in rows:
            values = []
            for col in selected_cols:
                value = row[col]
                if isinstance(value, float):
                    value = f"{value:.6f}"
                value = str(value).replace("|", "/")
                values.append(value)
            f.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()

    s2_root = Path(args.s2_root)
    output_dir = Path(args.output_dir)

    if not s2_root.exists():
        raise FileNotFoundError(f"S2 root does not exist: {s2_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    city_rasters = discover_city_rasters(s2_root, args.cities)

    print(f"[INFO] S2 root: {s2_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Cities to process: {len(city_rasters)}")
    print(f"[INFO] all_zero_as_nodata: {args.all_zero_as_nodata}")
    print(f"[INFO] nan_as_nodata: {args.nan_as_nodata}")
    print(f"[INFO] border_margin: {args.border_margin}")
    print(f"[INFO] internal_fill_threshold_percent: {args.internal_fill_threshold_percent}")
    print(f"[INFO] mostly_border_share_percent: {args.mostly_border_share_percent}")

    rows: list[dict] = []

    for i, (city, raster_path) in enumerate(city_rasters.items(), start=1):
        print(f"\n[STEP {i}/{len(city_rasters)}] {city}")
        print(f"[INFO] Raster: {raster_path}")

        try:
            row = inspect_city(city=city, raster_path=raster_path, args=args)
            rows.append(row)

            print(
                "[OK] "
                f"combined={row['combined_nodata_percent']:.4f}% | "
                f"border_share={row['border_share_percent']:.2f}% | "
                f"internal_share={row['internal_share_percent']:.2f}% | "
                f"internal_abs={row['internal_nodata_percent']:.4f}% | "
                f"action={row['recommended_action']}"
            )

        except Exception as exc:
            print(f"[ERROR] Failed city {city}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "city": city,
                    "recommended_action": "error",
                    "reason": str(exc),
                    "s2_path": str(raster_path),
                    "width": "",
                    "height": "",
                    "band_count": "",
                    "dtype": "",
                    "crs": "",
                    "raster_nodata_value": "",
                    "total_pixels": "",
                    "official_nodata_pixels": "",
                    "all_zero_allbands_pixels": "",
                    "nonfinite_pixels": "",
                    "combined_nodata_pixels": "",
                    "border_connected_nodata_pixels": "",
                    "internal_nodata_pixels": "",
                    "official_nodata_percent": "",
                    "all_zero_allbands_percent": "",
                    "nonfinite_percent": "",
                    "combined_nodata_percent": "",
                    "border_connected_nodata_percent": "",
                    "internal_nodata_percent": "",
                    "border_share_percent": "",
                    "internal_share_percent": "",
                    "border_margin_pixels": args.border_margin,
                    "all_zero_as_nodata": args.all_zero_as_nodata,
                    "nan_as_nodata": args.nan_as_nodata,
                }
            )

    rows = sorted(rows, key=lambda r: str(r["city"]))

    csv_path = output_dir / "city_s2_nodata_repair_plan.csv"
    json_path = output_dir / "city_s2_nodata_repair_plan.json"
    md_path = output_dir / "city_s2_nodata_repair_plan.md"

    write_csv(rows, csv_path, overwrite=args.overwrite)
    write_json(rows, json_path, overwrite=args.overwrite)
    write_markdown(rows, md_path, overwrite=args.overwrite)

    print("\n[DONE] Wrote:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    print("\n[SUMMARY]")
    action_counts: dict[str, int] = {}
    for row in rows:
        action = row["recommended_action"]
        action_counts[action] = action_counts.get(action, 0) + 1

    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count}")


if __name__ == "__main__":
    main()