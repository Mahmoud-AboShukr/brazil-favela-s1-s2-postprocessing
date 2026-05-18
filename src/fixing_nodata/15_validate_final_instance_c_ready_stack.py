#!/usr/bin/env python3
"""
Validate final repaired instance C ready stack.

This script validates the final city-level stack:

    instance_C_s2_nodata_repaired/
        s2_filled/
        s1_ready/
        labels/

It checks, per city:
    - S2-filled raster exists
    - S1-ready raster exists
    - label raster exists
    - S2 has expected 12 bands
    - S1 has expected 3 bands
    - label has expected 1 band
    - S2/S1/label grids are aligned
    - S2 nodata = 0
    - S1 nodata = 0
    - label values are binary 0/1
    - optional S2/S1 QA rasters exist and are aligned

It does not modify rasters.

Outputs:
    qc/final_ready_validation/
        final_instance_c_ready_stack_validation.csv
        final_instance_c_ready_stack_validation.json
        final_instance_c_ready_stack_validation.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate final instance C S2-filled/S1-ready/label stack."
    )

    parser.add_argument(
        "--instance-root",
        type=str,
        required=True,
        help="Root of instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--s2-subdir",
        type=str,
        default="s2_filled",
        help="S2-filled subdirectory. Default: s2_filled",
    )

    parser.add_argument(
        "--s1-subdir",
        type=str,
        default="s1_ready",
        help="S1-ready subdirectory. Default: s1_ready",
    )

    parser.add_argument(
        "--labels-subdir",
        type=str,
        default="labels",
        help="Labels subdirectory. Default: labels",
    )

    parser.add_argument(
        "--qc-subdir",
        type=str,
        default="qc/final_ready_validation",
        help="QC output subdirectory. Default: qc/final_ready_validation",
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional city list. If omitted, all cities under s2_filled are validated.",
    )

    parser.add_argument(
        "--expected-s2-bands",
        type=int,
        default=12,
        help="Expected S2 band count. Default: 12",
    )

    parser.add_argument(
        "--expected-s1-bands",
        type=int,
        default=3,
        help="Expected S1 band count. Default: 3",
    )

    parser.add_argument(
        "--expected-label-bands",
        type=int,
        default=1,
        help="Expected label band count. Default: 1",
    )

    s2_zero_group = parser.add_mutually_exclusive_group()
    s2_zero_group.add_argument(
        "--s2-all-zero-as-nodata",
        dest="s2_all_zero_as_nodata",
        action="store_true",
        help="Treat S2 all-zero-all-band pixels as nodata.",
    )
    s2_zero_group.add_argument(
        "--no-s2-all-zero-as-nodata",
        dest="s2_all_zero_as_nodata",
        action="store_false",
        help="Do not treat S2 all-zero-all-band pixels as nodata.",
    )
    parser.set_defaults(s2_all_zero_as_nodata=True)

    s1_zero_group = parser.add_mutually_exclusive_group()
    s1_zero_group.add_argument(
        "--s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_true",
        help="Treat S1 all-zero-all-band pixels as nodata.",
    )
    s1_zero_group.add_argument(
        "--no-s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_false",
        help="Do not treat S1 all-zero-all-band pixels as nodata.",
    )
    parser.set_defaults(s1_all_zero_as_nodata=False)

    nan_group = parser.add_mutually_exclusive_group()
    nan_group.add_argument(
        "--nan-as-nodata",
        dest="nan_as_nodata",
        action="store_true",
        help="Treat NaN/Inf as nodata.",
    )
    nan_group.add_argument(
        "--no-nan-as-nodata",
        dest="nan_as_nodata",
        action="store_false",
        help="Do not treat NaN/Inf as nodata.",
    )
    parser.set_defaults(nan_as_nodata=True)

    parser.add_argument(
        "--label-positive-threshold",
        type=float,
        default=0.5,
        help="Label pixels > this threshold are treated as positive. Default: 0.5",
    )

    parser.add_argument(
        "--fail-on-label-mask-nodata",
        action="store_true",
        help=(
            "Fail if label raster mask has nodata. Normally disabled because "
            "some binary labels may have nodata metadata equal to 0, where 0 is also background."
        ),
    )

    parser.add_argument(
        "--require-qa",
        action="store_true",
        help=(
            "Require S2 fill-level QA and S1 fill-source QA rasters to exist, "
            "be aligned, and contain expected values."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing validation outputs.",
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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_jsonable(v) for v in value]
    return str(value)


def discover_cities(s2_root: Path, cities: list[str] | None) -> list[str]:
    if cities:
        return sorted(cities)

    discovered = sorted([p.name for p in s2_root.iterdir() if p.is_dir()])

    if not discovered:
        raise FileNotFoundError(f"No city folders found under {s2_root}")

    return discovered


def find_s2_raster(instance_root: Path, s2_subdir: str, city: str) -> Path:
    city_dir = instance_root / s2_subdir / city

    candidates = sorted(city_dir.glob(f"{city}_s2_12bands_reflectance_10m.tif"))
    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s2*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No S2-filled raster found for {city}: {city_dir}")

    return candidates[0]


def find_s1_raster(instance_root: Path, s1_subdir: str, city: str) -> Path:
    city_dir = instance_root / s1_subdir / city

    candidates = sorted(city_dir.glob(f"{city}_s1_ready_vv_vh_vvdiff_10m_aligned.tif"))
    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s1_ready*.tif"))
    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s1*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No S1-ready raster found for {city}: {city_dir}")

    return candidates[0]


def find_label_raster(instance_root: Path, labels_subdir: str, city: str) -> Path:
    city_dir = instance_root / labels_subdir / city

    candidates = sorted(city_dir.glob(f"{city}_label_final.tif"))
    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_label*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No label raster found for {city}: {city_dir}")

    return candidates[0]


def find_s2_fill_level(instance_root: Path, city: str) -> Path | None:
    path = (
        instance_root
        / "qc"
        / "s2_fill"
        / "fill_level"
        / city
        / f"{city}_s2_fill_level.tif"
    )
    return path if path.exists() else None


def find_s1_fill_source(instance_root: Path, city: str) -> Path | None:
    path = (
        instance_root
        / "qc"
        / "s1_ready_merge"
        / "fill_source"
        / city
        / f"{city}_s1_fill_source.tif"
    )
    return path if path.exists() else None


def alignment_against_reference(reference_path: Path, other_path: Path, prefix: str) -> dict:
    with rasterio.open(reference_path) as ref, rasterio.open(other_path) as other:
        same_width = ref.width == other.width
        same_height = ref.height == other.height
        same_crs = ref.crs == other.crs
        same_transform = ref.transform.almost_equals(other.transform)

        return {
            f"{prefix}_same_width": same_width,
            f"{prefix}_same_height": same_height,
            f"{prefix}_same_crs": same_crs,
            f"{prefix}_same_transform": same_transform,
            f"{prefix}_alignment_ok": (
                same_width and same_height and same_crs and same_transform
            ),
        }


def raster_basic_info(path: Path, prefix: str) -> dict:
    with rasterio.open(path) as src:
        return {
            f"{prefix}_width": src.width,
            f"{prefix}_height": src.height,
            f"{prefix}_band_count": src.count,
            f"{prefix}_dtype": src.dtypes[0],
            f"{prefix}_crs": str(src.crs),
            f"{prefix}_transform": tuple(src.transform),
            f"{prefix}_nodata_value": src.nodata,
        }


def inspect_multiband_nodata(
    path: Path,
    all_zero_as_nodata: bool,
    nan_as_nodata: bool,
    prefix: str,
) -> dict:
    with rasterio.open(path) as src:
        height = src.height
        width = src.width
        count = src.count
        total = height * width

        combined_invalid = np.zeros((height, width), dtype=bool)
        official_invalid = np.zeros((height, width), dtype=bool)
        all_zero = np.zeros((height, width), dtype=bool)
        nonfinite = np.zeros((height, width), dtype=bool)

        indexes = list(range(1, count + 1))

        for _, window in src.block_windows(1):
            row0 = int(window.row_off)
            row1 = int(window.row_off + window.height)
            col0 = int(window.col_off)
            col1 = int(window.col_off + window.width)

            masks = src.read_masks(indexes=indexes, window=window)
            official_block = np.any(masks == 0, axis=0)

            block_invalid = official_block.copy()
            official_invalid[row0:row1, col0:col1] = official_block

            data = src.read(indexes=indexes, window=window)

            all_zero_block = np.all(data == 0, axis=0)
            all_zero[row0:row1, col0:col1] = all_zero_block

            nonfinite_block = np.any(~np.isfinite(data), axis=0)
            nonfinite[row0:row1, col0:col1] = nonfinite_block

            if all_zero_as_nodata:
                block_invalid |= all_zero_block

            if nan_as_nodata:
                block_invalid |= nonfinite_block

            combined_invalid[row0:row1, col0:col1] = block_invalid

        combined_pixels = int(combined_invalid.sum())
        official_pixels = int(official_invalid.sum())
        all_zero_pixels = int(all_zero.sum())
        nonfinite_pixels = int(nonfinite.sum())

        return {
            f"{prefix}_total_pixels": int(total),
            f"{prefix}_official_nodata_pixels": official_pixels,
            f"{prefix}_official_nodata_percent": percent(official_pixels, total),
            f"{prefix}_all_zero_allbands_pixels": all_zero_pixels,
            f"{prefix}_all_zero_allbands_percent": percent(all_zero_pixels, total),
            f"{prefix}_nonfinite_pixels": nonfinite_pixels,
            f"{prefix}_nonfinite_percent": percent(nonfinite_pixels, total),
            f"{prefix}_combined_nodata_pixels": combined_pixels,
            f"{prefix}_combined_nodata_percent": percent(combined_pixels, total),
            f"{prefix}_all_zero_as_nodata": all_zero_as_nodata,
            f"{prefix}_nan_as_nodata": nan_as_nodata,
        }


def inspect_label(label_path: Path, positive_threshold: float) -> dict:
    unique_values: set[float] = set()
    unique_overflow = False

    total_pixels = 0
    positive_pixels = 0
    mask_nodata_pixels = 0
    nonfinite_pixels = 0

    label_min = None
    label_max = None

    with rasterio.open(label_path) as src:
        for _, window in src.block_windows(1):
            data = src.read(1, window=window, masked=False)
            mask = src.read_masks(1, window=window)

            total_pixels += int(data.size)
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


def inspect_qa_raster(
    qa_path: Path | None,
    reference_path: Path,
    expected_values: set[int],
    required: bool,
    prefix: str,
) -> dict:
    if qa_path is None:
        return {
            f"{prefix}_path": "",
            f"{prefix}_exists": False,
            f"{prefix}_alignment_ok": False if required else "",
            f"{prefix}_unique_values": "",
            f"{prefix}_expected_values_ok": False if required else "",
            f"{prefix}_status": "missing_required" if required else "missing_optional",
        }

    with rasterio.open(reference_path) as ref, rasterio.open(qa_path) as qa:
        alignment_ok = (
            ref.width == qa.width
            and ref.height == qa.height
            and ref.crs == qa.crs
            and ref.transform.almost_equals(qa.transform)
            and qa.count == 1
        )

        unique_values: set[int] = set()
        counts: dict[int, int] = {}

        for _, window in qa.block_windows(1):
            data = qa.read(1, window=window, masked=False)
            values, value_counts = np.unique(data, return_counts=True)

            for value, count in zip(values, value_counts):
                ivalue = int(value)
                unique_values.add(ivalue)
                counts[ivalue] = counts.get(ivalue, 0) + int(count)

        expected_ok = unique_values.issubset(expected_values)
        values_repr = ",".join(str(v) for v in sorted(unique_values))

        if alignment_ok and expected_ok:
            status = "ok"
        elif not alignment_ok:
            status = "bad_alignment"
        else:
            status = "unexpected_values"

        row = {
            f"{prefix}_path": str(qa_path),
            f"{prefix}_exists": True,
            f"{prefix}_alignment_ok": alignment_ok,
            f"{prefix}_unique_values": values_repr,
            f"{prefix}_expected_values_ok": expected_ok,
            f"{prefix}_status": status,
        }

        for value in sorted(expected_values):
            row[f"{prefix}_count_value_{value}"] = counts.get(value, 0)

        return row


def classify_city(row: dict, args: argparse.Namespace) -> tuple[str, str]:
    if row.get("s2_band_count") != args.expected_s2_bands:
        return "fail_s2_band_count", f"S2 band count is {row.get('s2_band_count')}"

    if row.get("s1_band_count") != args.expected_s1_bands:
        return "fail_s1_band_count", f"S1 band count is {row.get('s1_band_count')}"

    if row.get("label_band_count") != args.expected_label_bands:
        return "fail_label_band_count", f"Label band count is {row.get('label_band_count')}"

    if not row.get("s1_alignment_ok"):
        return "fail_s1_alignment", "S1-ready is not aligned with S2-filled"

    if not row.get("label_alignment_ok"):
        return "fail_label_alignment", "Label raster is not aligned with S2-filled"

    if row.get("s2_combined_nodata_pixels", 1) != 0:
        return "fail_s2_has_nodata", "S2-filled still contains nodata"

    if row.get("s1_combined_nodata_pixels", 1) != 0:
        return "fail_s1_has_nodata", "S1-ready still contains nodata"

    if not row.get("label_binary_ok"):
        return "fail_label_not_binary", "Label raster contains values outside 0/1"

    if args.fail_on_label_mask_nodata and row.get("label_mask_nodata_pixels", 0) > 0:
        return "fail_label_mask_nodata", "Label raster mask contains nodata"

    if args.require_qa:
        if row.get("s2_fill_level_status") != "ok":
            return "fail_s2_fill_level_qa", "S2 fill-level QA missing or invalid"

        if row.get("s1_fill_source_status") != "ok":
            return "fail_s1_fill_source_qa", "S1 fill-source QA missing or invalid"

        if row.get("s1_fill_source_count_value_9", 0) > 0:
            return "fail_s1_unresolved_fill_source", "S1 fill-source QA contains unresolved value 9"

    return (
        "pass_final_ready_stack",
        "S2-filled, S1-ready, and labels are aligned; S2/S1 nodata-free; labels binary",
    )


def inspect_city(city: str, instance_root: Path, args: argparse.Namespace) -> dict:
    s2_path = find_s2_raster(instance_root, args.s2_subdir, city)
    s1_path = find_s1_raster(instance_root, args.s1_subdir, city)
    label_path = find_label_raster(instance_root, args.labels_subdir, city)

    row: dict[str, Any] = {
        "city": city,
        "s2_path": str(s2_path),
        "s1_path": str(s1_path),
        "label_path": str(label_path),
    }

    row.update(raster_basic_info(s2_path, "s2"))
    row.update(raster_basic_info(s1_path, "s1"))
    row.update(raster_basic_info(label_path, "label"))

    row.update(alignment_against_reference(s2_path, s1_path, "s1"))
    row.update(alignment_against_reference(s2_path, label_path, "label"))

    row["all_alignment_ok"] = bool(row["s1_alignment_ok"] and row["label_alignment_ok"])

    row.update(
        inspect_multiband_nodata(
            path=s2_path,
            all_zero_as_nodata=args.s2_all_zero_as_nodata,
            nan_as_nodata=args.nan_as_nodata,
            prefix="s2",
        )
    )

    row.update(
        inspect_multiband_nodata(
            path=s1_path,
            all_zero_as_nodata=args.s1_all_zero_as_nodata,
            nan_as_nodata=args.nan_as_nodata,
            prefix="s1",
        )
    )

    row.update(
        inspect_label(
            label_path=label_path,
            positive_threshold=args.label_positive_threshold,
        )
    )

    s2_fill_level_path = find_s2_fill_level(instance_root, city)
    s1_fill_source_path = find_s1_fill_source(instance_root, city)

    row.update(
        inspect_qa_raster(
            qa_path=s2_fill_level_path,
            reference_path=s2_path,
            expected_values={0, 3},
            required=args.require_qa,
            prefix="s2_fill_level",
        )
    )

    row.update(
        inspect_qa_raster(
            qa_path=s1_fill_source_path,
            reference_path=s1_path,
            expected_values={0, 1, 2, 9},
            required=args.require_qa,
            prefix="s1_fill_source",
        )
    )

    status, reason = classify_city(row, args)

    row["city_status"] = status
    row["city_status_reason"] = reason

    return row


def write_csv(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")

    if not rows:
        raise ValueError("No rows to write.")

    fields: list[str] = []
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


def write_json(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(safe_jsonable(rows), f, indent=2, ensure_ascii=False)


def write_markdown(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {path}")

    cols = [
        "city",
        "city_status",
        "all_alignment_ok",
        "s2_band_count",
        "s1_band_count",
        "label_band_count",
        "s2_combined_nodata_percent",
        "s1_combined_nodata_percent",
        "label_binary_ok",
        "label_positive_percent",
        "s2_fill_level_status",
        "s1_fill_source_status",
        "s1_fill_source_unique_values",
        "city_status_reason",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Final instance C ready-stack validation\n\n")
        f.write(
            "This report validates the final repaired city-level stack: "
            "`s2_filled`, `s1_ready`, and `labels`.\n\n"
        )

        f.write("## Status counts\n\n")
        counts: dict[str, int] = {}
        for row in rows:
            status = row.get("city_status", "unknown")
            counts[status] = counts.get(status, 0) + 1

        for status, count in sorted(counts.items()):
            f.write(f"- `{status}`: {count}\n")

        f.write("\n## Global checks\n\n")
        total_cities = len(rows)
        pass_count = sum(1 for row in rows if row.get("city_status") == "pass_final_ready_stack")
        s2_zero = sum(1 for row in rows if row.get("s2_combined_nodata_pixels") == 0)
        s1_zero = sum(1 for row in rows if row.get("s1_combined_nodata_pixels") == 0)
        binary = sum(1 for row in rows if row.get("label_binary_ok") is True)
        aligned = sum(1 for row in rows if row.get("all_alignment_ok") is True)

        f.write(f"- Cities validated: `{total_cities}`\n")
        f.write(f"- Passing cities: `{pass_count}`\n")
        f.write(f"- S2 nodata-free cities: `{s2_zero}`\n")
        f.write(f"- S1 nodata-free cities: `{s1_zero}`\n")
        f.write(f"- Binary-label cities: `{binary}`\n")
        f.write(f"- Fully aligned cities: `{aligned}`\n")

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

    s2_root = instance_root / args.s2_subdir
    s1_root = instance_root / args.s1_subdir
    labels_root = instance_root / args.labels_subdir
    qc_root = instance_root / args.qc_subdir

    if not s2_root.exists():
        raise FileNotFoundError(f"S2-filled root does not exist: {s2_root}")
    if not s1_root.exists():
        raise FileNotFoundError(f"S1-ready root does not exist: {s1_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels root does not exist: {labels_root}")

    qc_root.mkdir(parents=True, exist_ok=True)

    cities = discover_cities(s2_root, args.cities)

    print(f"[INFO] Instance root: {instance_root}")
    print(f"[INFO] S2-filled root: {s2_root}")
    print(f"[INFO] S1-ready root: {s1_root}")
    print(f"[INFO] Labels root: {labels_root}")
    print(f"[INFO] QC root: {qc_root}")
    print(f"[INFO] Cities to validate: {len(cities)}")
    print(f"[INFO] s2_all_zero_as_nodata: {args.s2_all_zero_as_nodata}")
    print(f"[INFO] s1_all_zero_as_nodata: {args.s1_all_zero_as_nodata}")
    print(f"[INFO] nan_as_nodata: {args.nan_as_nodata}")
    print(f"[INFO] require_qa: {args.require_qa}")
    print(f"[INFO] fail_on_label_mask_nodata: {args.fail_on_label_mask_nodata}")

    rows: list[dict] = []

    for idx, city in enumerate(cities, start=1):
        print(f"\n[STEP {idx}/{len(cities)}] {city}")

        try:
            row = inspect_city(city=city, instance_root=instance_root, args=args)
            rows.append(row)

            print(
                "[OK] "
                f"status={row['city_status']} | "
                f"alignment={row['all_alignment_ok']} | "
                f"S2 nodata={row['s2_combined_nodata_percent']:.6f}% | "
                f"S1 nodata={row['s1_combined_nodata_percent']:.6f}% | "
                f"label_binary={row['label_binary_ok']} | "
                f"S2_QA={row['s2_fill_level_status']} | "
                f"S1_QA={row['s1_fill_source_status']}"
            )

        except Exception as exc:
            print(f"[ERROR] {city}: {exc}")
            rows.append(
                {
                    "city": city,
                    "city_status": "error",
                    "city_status_reason": str(exc),
                }
            )

    rows = sorted(rows, key=lambda r: str(r.get("city", "")))

    csv_path = qc_root / "final_instance_c_ready_stack_validation.csv"
    json_path = qc_root / "final_instance_c_ready_stack_validation.json"
    md_path = qc_root / "final_instance_c_ready_stack_validation.md"

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
        status = row.get("city_status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()