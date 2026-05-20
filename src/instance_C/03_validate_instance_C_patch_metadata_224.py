#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_validate_instance_C_patch_metadata_224.py

Final validation gate for Instance C 224x224 patch metadata.

This script validates that the patch metadata produced by:

    01_build_patch_tiling_index_224.py
    02_compute_patch_metadata_224.py

is ready for RTC-vs-SNAP-GRD comparison and CROMA experiments.

It checks:

    1. Metadata CSV exists and has expected number of rows.
    2. Required columns exist.
    3. patch_id values are unique.
    4. Expected city count is correct.
    5. Expected positive / empty patch counts are correct.
    6. All source paths exist.
    7. S2/SNAP/RTC/label band-count contract is correct:
        S2        = 12 bands
        SNAP-GRD  = 3 bands
        RTC       = 2 bands
        label     = 1 band
    8. All S2/SNAP/RTC valid percentages are 100%.
    9. All RTC zero percentages are 0%.
    10. All labels are binary.
    11. patch_label_binary is consistent with label_positive_pixels.
    12. label_density_bin is consistent with label_positive_percent.
    13. Patch windows are inside city raster dimensions.
    14. Per-city raster stacks can be opened and are aligned.

Outputs:

    <instance-root>/metadata/instance_C_patches/
        patch_metadata_validation_checks_ps224_st112_cover.csv
        patch_metadata_validation_errors_ps224_st112_cover.csv
        patch_metadata_validation_city_ps224_st112_cover.csv
        patch_metadata_validation_ps224_st112_cover.json
        patch_metadata_validation_ps224_st112_cover.md

Example:

python src/instance_C/03_validate_instance_C_patch_metadata_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


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


def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    overwrite: bool,
    fieldnames: Optional[List[str]] = None,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    if fieldnames is None:
        if not rows:
            fail(f"No rows and no fieldnames provided for CSV: {path_to_str(path)}")
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
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
    city_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Instance C patch metadata validation")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Metadata CSV: `{summary['metadata_csv']}`")
    lines.append(f"- Total patches: `{summary['total_patches']}`")
    lines.append(f"- City count: `{summary['city_count']}`")
    lines.append(f"- Positive patches: `{summary['positive_patches']}`")
    lines.append(f"- Empty patches: `{summary['empty_patches']}`")
    lines.append(f"- Validation status: `{summary['validation_status']}`")
    lines.append(f"- Error count: `{summary['error_count']}`")
    lines.append(f"- Warning count: `{summary['warning_count']}`")
    lines.append("")

    lines.append("## Critical readiness indicators")
    lines.append("")
    lines.append(f"- S2 valid for all patches: `{summary['readiness']['s2_valid_all_patches']}`")
    lines.append(f"- SNAP-GRD valid for all patches: `{summary['readiness']['s1_snap_valid_all_patches']}`")
    lines.append(f"- RTC valid for all patches: `{summary['readiness']['s1_rtc_valid_all_patches']}`")
    lines.append(f"- RTC available for all patches: `{summary['readiness']['s1_rtc_available_all_patches']}`")
    lines.append(f"- RTC zero-free for all patches: `{summary['readiness']['s1_rtc_zero_free_all_patches']}`")
    lines.append(f"- Labels binary: `{summary['readiness']['labels_binary']}`")
    lines.append(f"- Source paths exist: `{summary['readiness']['source_paths_exist']}`")
    lines.append(f"- Raster stacks aligned: `{summary['readiness']['raster_stacks_aligned']}`")
    lines.append("")

    lines.append("## Validation checks")
    lines.append("")
    lines.append("| check | status | observed | expected | details |")
    lines.append("|---|---|---:|---:|---|")
    for row in checks:
        lines.append(
            f"| {row['check_name']} | "
            f"{row['status']} | "
            f"{row['observed']} | "
            f"{row['expected']} | "
            f"{row['details']} |"
        )

    lines.append("")
    lines.append("## City-level summary")
    lines.append("")
    lines.append(
        "| city | patches | positive | S2 min valid | SNAP min valid | RTC min valid | RTC max zero | status |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in city_rows:
        lines.append(
            f"| {row['city']} | "
            f"{row['patches']} | "
            f"{row['positive_patches']} | "
            f"{row['min_s2_valid_percent']} | "
            f"{row['min_s1_snap_valid_percent']} | "
            f"{row['min_s1_rtc_valid_percent']} | "
            f"{row['max_s1_rtc_zero_percent']} | "
            f"{row['status']} |"
        )

    lines.append("")
    lines.append("## Errors and warnings")
    lines.append("")

    if not errors:
        lines.append("No errors or warnings were detected.")
    else:
        lines.append("| severity | check | city | patch_id | message |")
        lines.append("|---|---|---|---|---|")
        for row in errors[:200]:
            lines.append(
                f"| {row['severity']} | "
                f"{row['check_name']} | "
                f"{row['city']} | "
                f"{row['patch_id']} | "
                f"{row['message']} |"
            )
        if len(errors) > 200:
            lines.append("")
            lines.append(f"Only the first 200 issues are shown here. Total issues: `{len(errors)}`.")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If validation status is `passed`, the Instance C patch metadata is ready for RTC-vs-SNAP-GRD comparison.")
    lines.append("- The main comparison should use identical patch IDs and labels for RTC and SNAP-GRD.")
    lines.append("- For the main CROMA comparison, use VV/VH from both RTC and SNAP-GRD. SNAP-GRD's VV-minus-VH band should not be used in the primary fair comparison.")
    lines.append("- This script does not modify data. It only validates metadata and source rasters.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def add_error(
    errors: List[Dict[str, object]],
    *,
    severity: str,
    check_name: str,
    city: str = "",
    patch_id: str = "",
    message: str,
) -> None:
    errors.append(
        {
            "severity": severity,
            "check_name": check_name,
            "city": city,
            "patch_id": patch_id,
            "message": message,
        }
    )


def add_check(
    checks: List[Dict[str, object]],
    *,
    check_name: str,
    passed: bool,
    observed: object,
    expected: object,
    details: str = "",
) -> None:
    checks.append(
        {
            "check_name": check_name,
            "status": "passed" if passed else "failed",
            "observed": observed,
            "expected": expected,
            "details": details,
        }
    )


def density_bin(label_positive_percent: float) -> str:
    value = float(label_positive_percent)

    if value <= 0.0:
        return "empty"
    if value < 1.0:
        return "low"
    if value < 10.0:
        return "medium"
    return "high"


def transforms_equal(a, b, tolerance: float) -> bool:
    aa = (
        float(a.a), float(a.b), float(a.c),
        float(a.d), float(a.e), float(a.f),
    )
    bb = (
        float(b.a), float(b.b), float(b.c),
        float(b.d), float(b.e), float(b.f),
    )

    return all(abs(x - y) <= tolerance for x, y in zip(aa, bb))


def group_rows_by_city(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[normalize_city(row["city"])].append(row)

    return dict(sorted(grouped.items()))


def as_path(value: str) -> Path:
    return Path(str(value).strip())


# ---------------------------------------------------------------------
# Main validation pieces
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "patch_id",
    "city",
    "region",
    "row_start",
    "col_start",
    "height",
    "width",
    "patch_size",
    "stride",
    "edge_mode",
    "city_width",
    "city_height",
    "source_s2_path",
    "source_s1_snap_path",
    "source_s1_rtc_path",
    "source_label_path",
    "s2_exists",
    "s1_snap_exists",
    "s1_rtc_exists",
    "label_exists",
    "s2_band_count",
    "s1_snap_band_count",
    "s1_rtc_band_count",
    "label_band_count",
    "s2_valid_percent",
    "s1_snap_valid_percent",
    "s1_rtc_valid_percent",
    "s1_rtc_zero_percent",
    "patch_label_binary",
    "label_positive_pixels",
    "label_positive_percent",
    "label_non_binary",
    "label_density_bin",
]


def validate_required_columns(
    rows: List[Dict[str, str]],
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
) -> None:
    columns = set(rows[0].keys())
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]

    add_check(
        checks,
        check_name="required_columns",
        passed=len(missing) == 0,
        observed=len(REQUIRED_COLUMNS) - len(missing),
        expected=len(REQUIRED_COLUMNS),
        details="" if not missing else "Missing: " + "; ".join(missing),
    )

    for col in missing:
        add_error(
            errors,
            severity="error",
            check_name="required_columns",
            message=f"Missing required column: {col}",
        )


def validate_global_counts(
    rows: List[Dict[str, str]],
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    total = len(rows)
    cities = sorted(set(normalize_city(r["city"]) for r in rows))
    positive = sum(1 for r in rows if safe_int(r["patch_label_binary"]) == 1)
    empty = total - positive

    checks_to_make = [
        ("total_patches", total, args.expected_total_patches),
        ("city_count", len(cities), args.expected_city_count),
        ("positive_patches", positive, args.expected_positive_patches),
        ("empty_patches", empty, args.expected_empty_patches),
    ]

    for check_name, observed, expected in checks_to_make:
        passed = observed == expected

        add_check(
            checks,
            check_name=check_name,
            passed=passed,
            observed=observed,
            expected=expected,
        )

        if not passed:
            add_error(
                errors,
                severity="error",
                check_name=check_name,
                message=f"{check_name}: observed {observed}, expected {expected}",
            )


def validate_unique_patch_ids(
    rows: List[Dict[str, str]],
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
) -> None:
    counter = Counter(r["patch_id"] for r in rows)
    duplicates = [patch_id for patch_id, count in counter.items() if count > 1]

    add_check(
        checks,
        check_name="unique_patch_ids",
        passed=len(duplicates) == 0,
        observed=len(duplicates),
        expected=0,
        details="" if not duplicates else "Duplicate sample: " + "; ".join(duplicates[:10]),
    )

    for patch_id in duplicates[:100]:
        add_error(
            errors,
            severity="error",
            check_name="unique_patch_ids",
            patch_id=patch_id,
            message=f"Duplicate patch_id appears {counter[patch_id]} times.",
        )


def validate_band_counts(
    rows: List[Dict[str, str]],
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    specs = [
        ("s2_band_count", args.expected_s2_bands),
        ("s1_snap_band_count", args.expected_s1_snap_bands),
        ("s1_rtc_band_count", args.expected_s1_rtc_bands),
        ("label_band_count", args.expected_label_bands),
    ]

    for col, expected in specs:
        bad = [
            r for r in rows
            if safe_int(r[col]) != int(expected)
        ]

        add_check(
            checks,
            check_name=f"{col}_contract",
            passed=len(bad) == 0,
            observed=f"{len(rows) - len(bad)}/{len(rows)}",
            expected=f"{len(rows)}/{len(rows)}",
            details=f"Expected {expected} for column {col}",
        )

        for r in bad[:100]:
            add_error(
                errors,
                severity="error",
                check_name=f"{col}_contract",
                city=normalize_city(r["city"]),
                patch_id=r["patch_id"],
                message=f"{col}={r[col]}, expected {expected}",
            )


def validate_availability_and_paths(
    rows: List[Dict[str, str]],
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
) -> None:
    bool_specs = [
        ("s2_exists", True),
        ("s1_snap_exists", True),
        ("s1_rtc_exists", True),
        ("label_exists", True),
        ("s1_rtc_available", True),
    ]

    for col, expected in bool_specs:
        if col not in rows[0]:
            add_error(
                errors,
                severity="error",
                check_name="availability_columns",
                message=f"Missing availability column: {col}",
            )
            continue

        bad = [r for r in rows if parse_bool(r[col]) != expected]

        add_check(
            checks,
            check_name=f"{col}_all_true",
            passed=len(bad) == 0,
            observed=f"{len(rows) - len(bad)}/{len(rows)}",
            expected=f"{len(rows)}/{len(rows)}",
        )

        for r in bad[:100]:
            add_error(
                errors,
                severity="error",
                check_name=f"{col}_all_true",
                city=normalize_city(r["city"]),
                patch_id=r["patch_id"],
                message=f"{col} is {r[col]}, expected True.",
            )

    path_cols = [
        "source_s2_path",
        "source_s1_snap_path",
        "source_s1_rtc_path",
        "source_label_path",
    ]

    unique_paths = {}

    for col in path_cols:
        unique_paths[col] = sorted(set(str(r[col]).strip() for r in rows if str(r[col]).strip()))

    missing_paths = []

    for col, paths in unique_paths.items():
        for p in paths:
            if not Path(p).exists():
                missing_paths.append((col, p))

    add_check(
        checks,
        check_name="source_paths_exist",
        passed=len(missing_paths) == 0,
        observed=sum(len(v) for v in unique_paths.values()) - len(missing_paths),
        expected=sum(len(v) for v in unique_paths.values()),
        details="" if not missing_paths else f"Missing paths: {len(missing_paths)}",
    )

    for col, p in missing_paths[:100]:
        add_error(
            errors,
            severity="error",
            check_name="source_paths_exist",
            message=f"{col} path does not exist: {p}",
        )


def validate_validity_percentages(
    rows: List[Dict[str, str]],
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    valid_cols = [
        "s2_valid_percent",
        "s1_snap_valid_percent",
        "s1_rtc_valid_percent",
    ]

    for col in valid_cols:
        bad = [
            r for r in rows
            if safe_float(r[col]) < 100.0 - float(args.percent_tolerance)
        ]

        add_check(
            checks,
            check_name=f"{col}_all_100",
            passed=len(bad) == 0,
            observed=f"{len(rows) - len(bad)}/{len(rows)}",
            expected=f"{len(rows)}/{len(rows)}",
        )

        for r in bad[:100]:
            add_error(
                errors,
                severity="error",
                check_name=f"{col}_all_100",
                city=normalize_city(r["city"]),
                patch_id=r["patch_id"],
                message=f"{col}={r[col]}, expected 100.",
            )

    rtc_zero_bad = [
        r for r in rows
        if safe_float(r["s1_rtc_zero_percent"]) > float(args.zero_percent_tolerance)
    ]

    add_check(
        checks,
        check_name="s1_rtc_zero_percent_all_0",
        passed=len(rtc_zero_bad) == 0,
        observed=f"{len(rows) - len(rtc_zero_bad)}/{len(rows)}",
        expected=f"{len(rows)}/{len(rows)}",
    )

    for r in rtc_zero_bad[:100]:
        add_error(
            errors,
            severity="error",
            check_name="s1_rtc_zero_percent_all_0",
            city=normalize_city(r["city"]),
            patch_id=r["patch_id"],
            message=f"s1_rtc_zero_percent={r['s1_rtc_zero_percent']}, expected 0.",
        )


def validate_labels(
    rows: List[Dict[str, str]],
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
) -> None:
    non_binary = [r for r in rows if parse_bool(r["label_non_binary"])]

    add_check(
        checks,
        check_name="labels_binary",
        passed=len(non_binary) == 0,
        observed=len(non_binary),
        expected=0,
    )

    for r in non_binary[:100]:
        add_error(
            errors,
            severity="error",
            check_name="labels_binary",
            city=normalize_city(r["city"]),
            patch_id=r["patch_id"],
            message="Patch label contains non-binary values.",
        )

    inconsistent_binary = []

    for r in rows:
        positive_pixels = safe_int(r["label_positive_pixels"])
        patch_label_binary = safe_int(r["patch_label_binary"])
        expected = 1 if positive_pixels > 0 else 0

        if patch_label_binary != expected:
            inconsistent_binary.append(r)

    add_check(
        checks,
        check_name="patch_label_binary_consistency",
        passed=len(inconsistent_binary) == 0,
        observed=len(inconsistent_binary),
        expected=0,
    )

    for r in inconsistent_binary[:100]:
        add_error(
            errors,
            severity="error",
            check_name="patch_label_binary_consistency",
            city=normalize_city(r["city"]),
            patch_id=r["patch_id"],
            message=(
                f"patch_label_binary={r['patch_label_binary']} but "
                f"label_positive_pixels={r['label_positive_pixels']}."
            ),
        )

    bad_density = []

    for r in rows:
        expected_bin = density_bin(safe_float(r["label_positive_percent"]))
        observed_bin = str(r["label_density_bin"])

        if observed_bin != expected_bin:
            bad_density.append((r, expected_bin))

    add_check(
        checks,
        check_name="label_density_bin_consistency",
        passed=len(bad_density) == 0,
        observed=len(bad_density),
        expected=0,
    )

    for r, expected_bin in bad_density[:100]:
        add_error(
            errors,
            severity="error",
            check_name="label_density_bin_consistency",
            city=normalize_city(r["city"]),
            patch_id=r["patch_id"],
            message=(
                f"label_density_bin={r['label_density_bin']}, "
                f"expected {expected_bin} from label_positive_percent={r['label_positive_percent']}."
            ),
        )


def validate_patch_windows(
    rows: List[Dict[str, str]],
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    bad = []

    for r in rows:
        row_start = safe_int(r["row_start"])
        col_start = safe_int(r["col_start"])
        height = safe_int(r["height"])
        width = safe_int(r["width"])
        city_height = safe_int(r["city_height"])
        city_width = safe_int(r["city_width"])
        patch_size = safe_int(r["patch_size"])
        stride = safe_int(r["stride"])

        ok = True

        if height != int(args.patch_size) or width != int(args.patch_size):
            ok = False

        if patch_size != int(args.patch_size):
            ok = False

        if stride != int(args.stride):
            ok = False

        if row_start < 0 or col_start < 0:
            ok = False

        if row_start + height > city_height:
            ok = False

        if col_start + width > city_width:
            ok = False

        if not ok:
            bad.append(r)

    add_check(
        checks,
        check_name="patch_windows_inside_city_bounds",
        passed=len(bad) == 0,
        observed=len(bad),
        expected=0,
    )

    for r in bad[:100]:
        add_error(
            errors,
            severity="error",
            check_name="patch_windows_inside_city_bounds",
            city=normalize_city(r["city"]),
            patch_id=r["patch_id"],
            message=(
                f"Window row={r['row_start']} col={r['col_start']} "
                f"h={r['height']} w={r['width']} city_h={r['city_height']} city_w={r['city_width']}."
            ),
        )


def validate_unique_city_paths(
    grouped: Dict[str, List[Dict[str, str]]],
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
) -> None:
    path_cols = [
        "source_s2_path",
        "source_s1_snap_path",
        "source_s1_rtc_path",
        "source_label_path",
    ]

    bad = []

    for city, rows in grouped.items():
        for col in path_cols:
            values = sorted(set(str(r[col]).strip() for r in rows))
            if len(values) != 1:
                bad.append((city, col, values))

    add_check(
        checks,
        check_name="one_source_path_per_city_per_modality",
        passed=len(bad) == 0,
        observed=len(bad),
        expected=0,
    )

    for city, col, values in bad[:100]:
        add_error(
            errors,
            severity="error",
            check_name="one_source_path_per_city_per_modality",
            city=city,
            message=f"{col} has {len(values)} unique paths in city. Values: {'; '.join(values[:5])}",
        )


def validate_raster_stacks(
    grouped: Dict[str, List[Dict[str, str]]],
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    bad = []

    for city, rows in grouped.items():
        first = rows[0]

        s2_path = as_path(first["source_s2_path"])
        snap_path = as_path(first["source_s1_snap_path"])
        rtc_path = as_path(first["source_s1_rtc_path"])
        label_path = as_path(first["source_label_path"])

        try:
            with rasterio.open(s2_path) as s2, \
                 rasterio.open(snap_path) as snap, \
                 rasterio.open(rtc_path) as rtc, \
                 rasterio.open(label_path) as label:

                city_errors = []

                if s2.count != int(args.expected_s2_bands):
                    city_errors.append(f"S2 bands={s2.count}, expected {args.expected_s2_bands}")

                if snap.count != int(args.expected_s1_snap_bands):
                    city_errors.append(f"SNAP bands={snap.count}, expected {args.expected_s1_snap_bands}")

                if rtc.count != int(args.expected_s1_rtc_bands):
                    city_errors.append(f"RTC bands={rtc.count}, expected {args.expected_s1_rtc_bands}")

                if label.count != int(args.expected_label_bands):
                    city_errors.append(f"Label bands={label.count}, expected {args.expected_label_bands}")

                ref_width = s2.width
                ref_height = s2.height
                ref_crs = s2.crs
                ref_transform = s2.transform

                for name, src in [
                    ("SNAP", snap),
                    ("RTC", rtc),
                    ("Label", label),
                ]:
                    if src.width != ref_width or src.height != ref_height:
                        city_errors.append(
                            f"{name} shape={src.width}x{src.height}, expected {ref_width}x{ref_height}"
                        )

                    if src.crs != ref_crs:
                        city_errors.append(f"{name} CRS mismatch")

                    if not transforms_equal(src.transform, ref_transform, float(args.transform_tolerance)):
                        city_errors.append(f"{name} transform mismatch")

                if city_errors:
                    bad.append((city, " | ".join(city_errors)))

        except Exception as exc:
            bad.append((city, repr(exc)))

    add_check(
        checks,
        check_name="raster_stacks_open_and_align",
        passed=len(bad) == 0,
        observed=len(grouped) - len(bad),
        expected=len(grouped),
    )

    for city, message in bad:
        add_error(
            errors,
            severity="error",
            check_name="raster_stacks_open_and_align",
            city=city,
            message=message,
        )


def summarize_cities(grouped: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, object]]:
    city_rows: List[Dict[str, object]] = []

    for city, rows in grouped.items():
        positives = sum(1 for r in rows if safe_int(r["patch_label_binary"]) == 1)
        s2_valids = [safe_float(r["s2_valid_percent"]) for r in rows]
        snap_valids = [safe_float(r["s1_snap_valid_percent"]) for r in rows]
        rtc_valids = [safe_float(r["s1_rtc_valid_percent"]) for r in rows]
        rtc_zeros = [safe_float(r["s1_rtc_zero_percent"]) for r in rows]

        city_rows.append(
            {
                "city": city,
                "region": rows[0].get("region", ""),
                "patches": len(rows),
                "positive_patches": positives,
                "min_s2_valid_percent": round(min(s2_valids), 8),
                "min_s1_snap_valid_percent": round(min(snap_valids), 8),
                "min_s1_rtc_valid_percent": round(min(rtc_valids), 8),
                "max_s1_rtc_zero_percent": round(max(rtc_zeros), 8),
                "status": "ok",
            }
        )

    return city_rows


def build_summary(
    *,
    instance_root: Path,
    metadata_csv: Path,
    rows: List[Dict[str, str]],
    checks: List[Dict[str, object]],
    errors: List[Dict[str, object]],
    city_rows: List[Dict[str, object]],
    args: argparse.Namespace,
    output_paths: Dict[str, Path],
) -> Dict[str, object]:
    total = len(rows)
    positives = sum(1 for r in rows if safe_int(r["patch_label_binary"]) == 1)
    empty = total - positives
    cities = sorted(set(normalize_city(r["city"]) for r in rows))

    error_count = sum(1 for e in errors if e["severity"] == "error")
    warning_count = sum(1 for e in errors if e["severity"] == "warning")

    check_status = {row["check_name"]: row["status"] for row in checks}

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "metadata_csv": path_to_str(metadata_csv),
        "total_patches": total,
        "city_count": len(cities),
        "positive_patches": positives,
        "empty_patches": empty,
        "validation_status": "passed" if error_count == 0 else "failed",
        "error_count": error_count,
        "warning_count": warning_count,
        "readiness": {
            "s2_valid_all_patches": check_status.get("s2_valid_percent_all_100") == "passed",
            "s1_snap_valid_all_patches": check_status.get("s1_snap_valid_percent_all_100") == "passed",
            "s1_rtc_valid_all_patches": check_status.get("s1_rtc_valid_percent_all_100") == "passed",
            "s1_rtc_available_all_patches": check_status.get("s1_rtc_available_all_true") == "passed",
            "s1_rtc_zero_free_all_patches": check_status.get("s1_rtc_zero_percent_all_0") == "passed",
            "labels_binary": check_status.get("labels_binary") == "passed",
            "source_paths_exist": check_status.get("source_paths_exist") == "passed",
            "raster_stacks_aligned": check_status.get("raster_stacks_open_and_align") == "passed",
        },
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "expected_total_patches": args.expected_total_patches,
            "expected_city_count": args.expected_city_count,
            "expected_positive_patches": args.expected_positive_patches,
            "expected_empty_patches": args.expected_empty_patches,
            "expected_s2_bands": args.expected_s2_bands,
            "expected_s1_snap_bands": args.expected_s1_snap_bands,
            "expected_s1_rtc_bands": args.expected_s1_rtc_bands,
            "expected_label_bands": args.expected_label_bands,
            "percent_tolerance": args.percent_tolerance,
            "zero_percent_tolerance": args.zero_percent_tolerance,
            "transform_tolerance": args.transform_tolerance,
            "check_rasters": not args.no_raster_check,
        },
        "outputs": {key: path_to_str(value) for key, value in output_paths.items()},
        "checks": checks,
        "errors": errors,
        "city_rows": city_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Instance C 224x224 patch metadata before CROMA comparison."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Patch metadata CSV. Default: <instance-root>/metadata/instance_C_patches/patch_metadata_ps<patch-size>_st<stride>_<edge-mode>.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <instance-root>/metadata/instance_C_patches.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size. Default: 224.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=112,
        help="Stride. Default: 112.",
    )

    parser.add_argument(
        "--edge-mode",
        choices=["cover", "drop"],
        default="cover",
        help="Edge mode. Default: cover.",
    )

    parser.add_argument(
        "--expected-total-patches",
        type=int,
        default=12699,
        help="Expected total patch count. Default: 12699.",
    )

    parser.add_argument(
        "--expected-city-count",
        type=int,
        default=26,
        help="Expected city count. Default: 26.",
    )

    parser.add_argument(
        "--expected-positive-patches",
        type=int,
        default=6382,
        help="Expected positive patch count. Default: 6382.",
    )

    parser.add_argument(
        "--expected-empty-patches",
        type=int,
        default=6317,
        help="Expected empty patch count. Default: 6317.",
    )

    parser.add_argument(
        "--expected-s2-bands",
        type=int,
        default=12,
        help="Expected S2 band count. Default: 12.",
    )

    parser.add_argument(
        "--expected-s1-snap-bands",
        type=int,
        default=3,
        help="Expected S1 SNAP-GRD band count. Default: 3.",
    )

    parser.add_argument(
        "--expected-s1-rtc-bands",
        type=int,
        default=2,
        help="Expected S1 RTC band count. Default: 2.",
    )

    parser.add_argument(
        "--expected-label-bands",
        type=int,
        default=1,
        help="Expected label band count. Default: 1.",
    )

    parser.add_argument(
        "--percent-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for 100 percent validity checks. Default: 1e-6.",
    )

    parser.add_argument(
        "--zero-percent-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for RTC zero percent checks. Default: 1e-6.",
    )

    parser.add_argument(
        "--transform-tolerance",
        type=float,
        default=0.0,
        help="Affine transform tolerance. Default: 0.0 exact.",
    )

    parser.add_argument(
        "--no-raster-check",
        action="store_true",
        help="Skip opening per-city source rasters for alignment validation.",
    )

    parser.add_argument(
        "--no-fail-on-error",
        action="store_true",
        help="Do not exit with non-zero status when validation errors are found.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite validation outputs.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root
    output_dir: Path = args.output_dir or (instance_root / "metadata" / "instance_C_patches")

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    metadata_csv: Path = args.metadata_csv or (
        output_dir / f"patch_metadata_{stem}.csv"
    )

    checks_csv = output_dir / f"patch_metadata_validation_checks_{stem}.csv"
    errors_csv = output_dir / f"patch_metadata_validation_errors_{stem}.csv"
    city_csv = output_dir / f"patch_metadata_validation_city_{stem}.csv"
    json_path = output_dir / f"patch_metadata_validation_{stem}.json"
    md_path = output_dir / f"patch_metadata_validation_{stem}.md"

    output_paths = {
        "checks_csv": checks_csv,
        "errors_csv": errors_csv,
        "city_csv": city_csv,
        "json": json_path,
        "markdown": md_path,
    }

    log("STEP", "Validating Instance C patch metadata.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Metadata CSV:  {path_to_str(metadata_csv)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    rows = read_csv_rows(metadata_csv)

    checks: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    validate_required_columns(rows, checks, errors)

    if any(e["check_name"] == "required_columns" and e["severity"] == "error" for e in errors):
        fail("Required columns are missing; cannot continue validation.")

    grouped = group_rows_by_city(rows)

    validate_global_counts(rows, checks, errors, args)
    validate_unique_patch_ids(rows, checks, errors)
    validate_band_counts(rows, checks, errors, args)
    validate_availability_and_paths(rows, checks, errors)
    validate_validity_percentages(rows, checks, errors, args)
    validate_labels(rows, checks, errors)
    validate_patch_windows(rows, checks, errors, args)
    validate_unique_city_paths(grouped, checks, errors)

    if not args.no_raster_check:
        validate_raster_stacks(grouped, checks, errors, args)

    city_rows = summarize_cities(grouped)

    summary = build_summary(
        instance_root=instance_root,
        metadata_csv=metadata_csv,
        rows=rows,
        checks=checks,
        errors=errors,
        city_rows=city_rows,
        args=args,
        output_paths=output_paths,
    )

    errors_for_csv = errors
    if not errors_for_csv:
        errors_for_csv = [
            {
                "severity": "none",
                "check_name": "all_checks",
                "city": "",
                "patch_id": "",
                "message": "No errors or warnings detected.",
            }
        ]

    log("STEP", "Writing validation outputs.")

    write_csv(
        checks_csv,
        checks,
        overwrite=bool(args.overwrite),
        fieldnames=["check_name", "status", "observed", "expected", "details"],
    )

    write_csv(
        errors_csv,
        errors_for_csv,
        overwrite=bool(args.overwrite),
        fieldnames=["severity", "check_name", "city", "patch_id", "message"],
    )

    write_csv(
        city_csv,
        city_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "city",
            "region",
            "patches",
            "positive_patches",
            "min_s2_valid_percent",
            "min_s1_snap_valid_percent",
            "min_s1_rtc_valid_percent",
            "max_s1_rtc_zero_percent",
            "status",
        ],
    )

    write_json(json_path, summary, overwrite=bool(args.overwrite))
    write_markdown(md_path, summary, checks, errors, city_rows, overwrite=bool(args.overwrite))

    log("OK", f"Wrote checks CSV: {path_to_str(checks_csv)}")
    log("OK", f"Wrote errors CSV: {path_to_str(errors_csv)}")
    log("OK", f"Wrote city CSV:   {path_to_str(city_csv)}")
    log("OK", f"Wrote JSON:       {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:   {path_to_str(md_path)}")

    log("STEP", "Final validation summary.")
    log("OK" if summary["validation_status"] == "passed" else "ERROR", f"Validation status: {summary['validation_status']}")
    log("OK", f"Total patches: {summary['total_patches']}")
    log("OK", f"City count: {summary['city_count']}")
    log("OK", f"Positive patches: {summary['positive_patches']}")
    log("OK", f"Empty patches: {summary['empty_patches']}")
    log("OK", f"Error count: {summary['error_count']}")
    log("OK", f"Warning count: {summary['warning_count']}")

    for key, value in summary["readiness"].items():
        log("OK" if value else "ERROR", f"{key}: {value}")

    if summary["error_count"] > 0 and not args.no_fail_on_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()