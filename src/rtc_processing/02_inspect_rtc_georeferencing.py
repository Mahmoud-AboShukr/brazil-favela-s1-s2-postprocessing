#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_inspect_rtc_georeferencing.py

Inspect georeferencing quality of the RTC candidate files selected by:

    src/rtc_processing/01_inventory_s1_rtc_raw.py

This script DOES NOT write or modify rasters.

It answers:

    1. Does each recommended RTC input have a normal CRS + transform?
    2. If CRS is missing, does it have GCPs?
    3. Can each city be safely aligned to the Instance C S2 grid?
    4. Should finalization use normal reprojection or GCP-based reprojection?
    5. Which cities would fail if we tried to finalize RTC now?

Expected inputs:

    <instance-root>/metadata/rtc_processing/raw_rtc_inventory_by_city.csv

Expected outputs:

    <instance-root>/metadata/rtc_processing/rtc_georeferencing_files.csv
    <instance-root>/metadata/rtc_processing/rtc_georeferencing_by_city.csv
    <instance-root>/metadata/rtc_processing/rtc_georeferencing.json
    <instance-root>/metadata/rtc_processing/rtc_georeferencing.md

Example PowerShell command:

python src/rtc_processing/02_inspect_rtc_georeferencing.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --overwrite

Optional strict mode:

python src/rtc_processing/02_inspect_rtc_georeferencing.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --fail-if-unusable `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import rasterio
    from rasterio.transform import Affine, from_gcps
except ImportError as exc:
    raise SystemExit(
        "[ERROR] rasterio is required but is not installed.\n"
        "Install it first, for example:\n"
        "    pip install rasterio\n\n"
        f"Original error: {exc}"
    )


# ---------------------------------------------------------------------
# Logging helpers
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
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def parse_bool_text(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


# ---------------------------------------------------------------------
# CSV / JSON / Markdown I/O
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
    city_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# RTC georeferencing inspection")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Input inventory: `{summary['input_city_inventory_csv']}`")
    lines.append(f"- Cities inspected: `{summary['n_cities_inspected']}`")
    lines.append(f"- Files inspected: `{summary['n_files_inspected']}`")
    lines.append(f"- Cities usable for finalization: `{summary['n_cities_usable_for_finalization']}`")
    lines.append(f"- Cities not usable yet: `{summary['n_cities_not_usable_yet']}`")
    lines.append("")

    lines.append("## Georeferencing routes")
    lines.append("")
    lines.append("| route | cities |")
    lines.append("|---|---:|")
    for route, count in summary["finalization_route_counts"].items():
        lines.append(f"| {route} | {count} |")
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    outputs = summary["outputs"]
    lines.append(f"- File report CSV: `{outputs['file_report_csv']}`")
    lines.append(f"- City report CSV: `{outputs['city_report_csv']}`")
    lines.append(f"- JSON: `{outputs['json']}`")
    lines.append(f"- Markdown: `{outputs['markdown']}`")
    lines.append("")

    lines.append("## City-level georeferencing status")
    lines.append("")
    lines.append(
        "| city | input mode | usable | route | stacked georef | VV georef | VH georef | notes |"
    )
    lines.append("|---|---|---:|---|---|---|---|---|")

    for row in city_rows:
        lines.append(
            f"| {row['city']} | "
            f"{row['recommended_mode']} | "
            f"{row['usable_for_finalization']} | "
            f"{row['finalization_route']} | "
            f"{row['stacked_georef_mode']} | "
            f"{row['vv_georef_mode']} | "
            f"{row['vh_georef_mode']} | "
            f"{row['notes']} |"
        )

    lines.append("")
    lines.append("## Cities not usable yet")
    lines.append("")

    bad_rows = [row for row in city_rows if not bool(row["usable_for_finalization"])]

    if bad_rows:
        for row in bad_rows:
            lines.append(f"- `{row['city']}`: {row['notes']}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `normal_crs_transform` means the raster has a CRS and a meaningful affine transform.")
    lines.append("- `gcps` means the raster lacks normal georeferencing but has Ground Control Points that may support GCP-based reprojection.")
    lines.append("- `insufficient_georef` means the raster does not have enough georeferencing information to align safely.")
    lines.append("- For final RTC integration, all cities must become 2-band VV/VH rasters aligned exactly to the S2 grid.")
    lines.append("- If a city is usable through GCPs, the next finalization script should reproject from GCPs to the S2 reference grid.")
    lines.append("- If a city is not usable, we need either better georeferenced RTC files or reprocessing/export from SNAP/RTC.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Georeferencing inspection
# ---------------------------------------------------------------------

def affine_to_tuple(transform: Affine) -> Tuple[float, float, float, float, float, float]:
    return tuple(float(x) for x in transform)


def is_identity_like_transform(transform: Affine, tolerance: float = 1e-9) -> bool:
    """
    Detect a default/identity-like transform.

    Rasterio often reports non-georeferenced rasters as:

        | 1, 0, 0 |
        | 0, 1, 0 |
        | 0, 0, 1 |

    For north-up projected geospatial rasters, the y pixel size is usually negative,
    e.g. 10, -10. A 1, +1 transform with origin at 0,0 is suspicious.
    """

    a, b, c, d, e, f = affine_to_tuple(transform)

    return (
        abs(a - 1.0) <= tolerance
        and abs(b) <= tolerance
        and abs(c) <= tolerance
        and abs(d) <= tolerance
        and abs(e - 1.0) <= tolerance
        and abs(f) <= tolerance
    )


def is_suspicious_transform(transform: Affine, crs_exists: bool) -> bool:
    """
    Suspicious means it likely does not encode real world position.

    This is conservative:
        - identity-like transform is suspicious
        - missing CRS + pixel size 1 is suspicious
    """

    a, b, c, d, e, f = affine_to_tuple(transform)

    if is_identity_like_transform(transform):
        return True

    if not crs_exists and abs(a) == 1.0 and abs(e) == 1.0:
        return True

    return False


def gcp_to_dict(gcp) -> Dict[str, object]:
    return {
        "row": float(gcp.row),
        "col": float(gcp.col),
        "x": float(gcp.x),
        "y": float(gcp.y),
        "z": float(gcp.z) if gcp.z is not None else 0.0,
        "id": "" if gcp.id is None else str(gcp.id),
        "info": "" if gcp.info is None else str(gcp.info),
    }


def inspect_single_raster(path: Optional[Path], role: str) -> Dict[str, object]:
    """
    Inspect CRS, transform, and GCP availability for one raster.
    """

    result: Dict[str, object] = {
        "role": role,
        "file_path": path_to_str(path),
        "exists": False,
        "open_status": "not_started",
        "open_error": "",
        "band_count": "",
        "width": "",
        "height": "",
        "dtypes": "",
        "nodata": "",
        "crs": "",
        "has_crs": False,
        "transform": "",
        "pixel_width": "",
        "pixel_height": "",
        "bounds_left": "",
        "bounds_bottom": "",
        "bounds_right": "",
        "bounds_top": "",
        "transform_identity_like": "",
        "transform_suspicious": "",
        "gcp_count": 0,
        "gcp_crs": "",
        "has_gcps": False,
        "gcp_transform_status": "not_attempted",
        "gcp_transform": "",
        "band_descriptions": "",
        "georef_mode": "missing_file",
        "can_align": False,
        "notes": "",
    }

    if path is None:
        result["notes"] = "No path provided."
        return result

    if not path.exists():
        result["notes"] = "File path does not exist."
        return result

    result["exists"] = True

    try:
        with rasterio.open(path) as src:
            result["open_status"] = "ok"
            result["band_count"] = src.count
            result["width"] = src.width
            result["height"] = src.height
            result["dtypes"] = ";".join(str(x) for x in src.dtypes)
            result["nodata"] = "" if src.nodata is None else src.nodata

            crs_exists = src.crs is not None
            result["has_crs"] = crs_exists
            result["crs"] = str(src.crs) if src.crs is not None else ""

            transform = src.transform
            result["transform"] = affine_to_tuple(transform)
            result["pixel_width"] = float(transform.a)
            result["pixel_height"] = float(transform.e)
            result["bounds_left"] = float(src.bounds.left)
            result["bounds_bottom"] = float(src.bounds.bottom)
            result["bounds_right"] = float(src.bounds.right)
            result["bounds_top"] = float(src.bounds.top)

            identity_like = is_identity_like_transform(transform)
            suspicious = is_suspicious_transform(transform, crs_exists=crs_exists)

            result["transform_identity_like"] = identity_like
            result["transform_suspicious"] = suspicious

            result["band_descriptions"] = ";".join(
                "" if desc is None else str(desc)
                for desc in src.descriptions
            )

            gcps, gcp_crs = src.gcps
            result["gcp_count"] = len(gcps)
            result["has_gcps"] = len(gcps) > 0
            result["gcp_crs"] = str(gcp_crs) if gcp_crs is not None else ""

            if len(gcps) > 0:
                try:
                    gcp_transform = from_gcps(gcps)
                    result["gcp_transform_status"] = "ok"
                    result["gcp_transform"] = affine_to_tuple(gcp_transform)
                except Exception as exc:
                    result["gcp_transform_status"] = f"failed: {repr(exc)}"

            normal_georef = crs_exists and not suspicious
            gcp_georef = len(gcps) >= 4 and gcp_crs is not None

            if normal_georef:
                result["georef_mode"] = "normal_crs_transform"
                result["can_align"] = True
                result["notes"] = "Usable through normal CRS/transform reprojection."
            elif gcp_georef:
                result["georef_mode"] = "gcps"
                result["can_align"] = True
                result["notes"] = "Usable through GCP-based reprojection."
            else:
                result["georef_mode"] = "insufficient_georef"
                result["can_align"] = False
                result["notes"] = "Missing usable CRS/transform and insufficient GCPs."

    except Exception as exc:
        result["open_status"] = "failed"
        result["open_error"] = repr(exc)
        result["georef_mode"] = "open_failed"
        result["can_align"] = False
        result["notes"] = f"Rasterio failed to open file: {repr(exc)}"

    return result


def compare_file_pair(
    vv: Dict[str, object],
    vh: Dict[str, object],
) -> Dict[str, object]:
    """
    Compare VV and VH files for separate_vv_vh mode.
    """

    notes: List[str] = []

    same_shape = (
        vv.get("width") == vh.get("width")
        and vv.get("height") == vh.get("height")
    )

    same_crs = vv.get("crs", "") == vh.get("crs", "")
    same_transform = vv.get("transform", "") == vh.get("transform", "")
    same_gcp_crs = vv.get("gcp_crs", "") == vh.get("gcp_crs", "")
    same_gcp_count = vv.get("gcp_count", 0) == vh.get("gcp_count", 0)

    if not same_shape:
        notes.append("VV/VH shapes differ.")

    if vv.get("georef_mode") == "normal_crs_transform" and vh.get("georef_mode") == "normal_crs_transform":
        if not same_crs:
            notes.append("VV/VH CRS differ.")
        if not same_transform:
            notes.append("VV/VH transforms differ.")

    if vv.get("georef_mode") == "gcps" and vh.get("georef_mode") == "gcps":
        if not same_gcp_crs:
            notes.append("VV/VH GCP CRS differ.")
        if not same_gcp_count:
            notes.append("VV/VH GCP counts differ.")

    pair_ok = (
        bool(vv.get("can_align"))
        and bool(vh.get("can_align"))
        and same_shape
    )

    return {
        "same_shape": same_shape,
        "same_crs": same_crs,
        "same_transform": same_transform,
        "same_gcp_crs": same_gcp_crs,
        "same_gcp_count": same_gcp_count,
        "pair_ok": pair_ok,
        "notes": " | ".join(notes),
    }


# ---------------------------------------------------------------------
# City-level inspection
# ---------------------------------------------------------------------

def path_from_cell(value: object) -> Optional[Path]:
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    return Path(text)


def inspect_city_row(row: Dict[str, str]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    city = row["city"]
    mode = row.get("recommended_mode", "")

    stacked_path = path_from_cell(row.get("recommended_stacked_path", ""))
    vv_path = path_from_cell(row.get("recommended_vv_path", ""))
    vh_path = path_from_cell(row.get("recommended_vh_path", ""))

    file_reports: List[Dict[str, object]] = []

    stacked_report = inspect_single_raster(stacked_path, role="stacked") if stacked_path else None
    vv_report = inspect_single_raster(vv_path, role="vv") if vv_path else None
    vh_report = inspect_single_raster(vh_path, role="vh") if vh_path else None

    for report in [stacked_report, vv_report, vh_report]:
        if report is not None:
            report["city"] = city
            report["recommended_mode"] = mode
            file_reports.append(report)

    notes: List[str] = []
    usable = False
    finalization_route = "not_usable_yet"

    stacked_georef_mode = ""
    vv_georef_mode = ""
    vh_georef_mode = ""

    if stacked_report is not None:
        stacked_georef_mode = str(stacked_report["georef_mode"])

    if vv_report is not None:
        vv_georef_mode = str(vv_report["georef_mode"])

    if vh_report is not None:
        vh_georef_mode = str(vh_report["georef_mode"])

    if mode == "stacked":
        if stacked_report is None:
            notes.append("Inventory recommends stacked mode but no stacked path is available.")
        elif safe_int(stacked_report.get("band_count", 0)) < 2:
            notes.append("Stacked file has fewer than 2 bands.")
        elif bool(stacked_report.get("can_align", False)):
            usable = True

            if stacked_report["georef_mode"] == "normal_crs_transform":
                finalization_route = "stacked_normal_reproject"
            elif stacked_report["georef_mode"] == "gcps":
                finalization_route = "stacked_gcp_reproject"
            else:
                finalization_route = "stacked_unknown_reproject"

            notes.append(str(stacked_report["notes"]))
        else:
            notes.append(str(stacked_report["notes"]))

    elif mode == "separate_vv_vh":
        if vv_report is None or vh_report is None:
            notes.append("Inventory recommends separate VV/VH mode but one or both paths are missing.")
        else:
            pair = compare_file_pair(vv_report, vh_report)

            if not bool(pair["pair_ok"]):
                notes.append("VV/VH pair is not safely alignable.")
                if pair["notes"]:
                    notes.append(str(pair["notes"]))

            elif vv_report["georef_mode"] == "normal_crs_transform" and vh_report["georef_mode"] == "normal_crs_transform":
                usable = True
                finalization_route = "separate_vv_vh_normal_reproject"
                notes.append("VV/VH pair usable through normal CRS/transform reprojection.")

            elif vv_report["georef_mode"] == "gcps" and vh_report["georef_mode"] == "gcps":
                usable = True
                finalization_route = "separate_vv_vh_gcp_reproject"
                notes.append("VV/VH pair usable through GCP-based reprojection.")

            else:
                # Mixed normal/GCP is technically possible but risky for finalization.
                usable = False
                finalization_route = "mixed_georef_manual_review"
                notes.append(
                    "VV/VH pair has mixed georeferencing modes. Manual review recommended."
                )

    else:
        notes.append(f"Unsupported or unclear recommended_mode: {mode}")

    city_report: Dict[str, object] = {
        "city": city,
        "recommended_mode": mode,
        "usable_for_finalization": usable,
        "finalization_route": finalization_route,
        "stacked_path": "" if stacked_path is None else path_to_str(stacked_path),
        "vv_path": "" if vv_path is None else path_to_str(vv_path),
        "vh_path": "" if vh_path is None else path_to_str(vh_path),
        "stacked_georef_mode": stacked_georef_mode,
        "vv_georef_mode": vv_georef_mode,
        "vh_georef_mode": vh_georef_mode,
        "stacked_gcp_count": stacked_report.get("gcp_count", "") if stacked_report else "",
        "vv_gcp_count": vv_report.get("gcp_count", "") if vv_report else "",
        "vh_gcp_count": vh_report.get("gcp_count", "") if vh_report else "",
        "stacked_crs": stacked_report.get("crs", "") if stacked_report else "",
        "vv_crs": vv_report.get("crs", "") if vv_report else "",
        "vh_crs": vh_report.get("crs", "") if vh_report else "",
        "notes": " | ".join(note for note in notes if note),
    }

    return file_reports, city_report


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect RTC candidate georeferencing before finalizing s1_rtc_ready."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--input-city-inventory",
        type=Path,
        default=None,
        help=(
            "Optional raw RTC city inventory CSV. "
            "Default: <instance-root>/metadata/rtc_processing/raw_rtc_inventory_by_city.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. "
            "Default: <instance-root>/metadata/rtc_processing"
        ),
    )

    parser.add_argument(
        "--fail-if-unusable",
        action="store_true",
        help="Exit with code 2 if any city is not usable for finalization.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root

    output_dir: Path = args.output_dir or (
        instance_root / "metadata" / "rtc_processing"
    )

    input_city_inventory: Path = args.input_city_inventory or (
        output_dir / "raw_rtc_inventory_by_city.csv"
    )

    file_report_csv = output_dir / "rtc_georeferencing_files.csv"
    city_report_csv = output_dir / "rtc_georeferencing_by_city.csv"
    json_path = output_dir / "rtc_georeferencing.json"
    md_path = output_dir / "rtc_georeferencing.md"

    log("STEP", "Inspecting RTC georeferencing.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Input inventory: {path_to_str(input_city_inventory)}")
    log("INFO", f"Output dir: {path_to_str(output_dir)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    city_inventory_rows = read_csv_rows(input_city_inventory)

    all_file_reports: List[Dict[str, object]] = []
    city_reports: List[Dict[str, object]] = []

    for row in city_inventory_rows:
        city = row["city"]
        log("STEP", f"Inspecting city: {city}")

        file_reports, city_report = inspect_city_row(row)

        all_file_reports.extend(file_reports)
        city_reports.append(city_report)

        status = "OK" if bool(city_report["usable_for_finalization"]) else "WARN"

        log(
            status,
            f"{city}: usable={city_report['usable_for_finalization']}, "
            f"route={city_report['finalization_route']}",
        )

    route_counts = Counter(str(row["finalization_route"]) for row in city_reports)

    n_usable = sum(1 for row in city_reports if bool(row["usable_for_finalization"]))
    n_not_usable = len(city_reports) - n_usable

    summary: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "input_city_inventory_csv": path_to_str(input_city_inventory),
        "n_cities_inspected": len(city_reports),
        "n_files_inspected": len(all_file_reports),
        "n_cities_usable_for_finalization": n_usable,
        "n_cities_not_usable_yet": n_not_usable,
        "finalization_route_counts": dict(sorted(route_counts.items())),
        "outputs": {
            "file_report_csv": path_to_str(file_report_csv),
            "city_report_csv": path_to_str(city_report_csv),
            "json": path_to_str(json_path),
            "markdown": path_to_str(md_path),
        },
        "city_reports": city_reports,
    }

    log("STEP", "Writing georeferencing reports.")

    write_csv(file_report_csv, all_file_reports, overwrite=args.overwrite)
    write_csv(city_report_csv, city_reports, overwrite=args.overwrite)
    write_json(json_path, summary, overwrite=args.overwrite)
    write_markdown(md_path, summary, city_reports, overwrite=args.overwrite)

    log("OK", f"Wrote file report CSV: {path_to_str(file_report_csv)}")
    log("OK", f"Wrote city report CSV: {path_to_str(city_report_csv)}")
    log("OK", f"Wrote JSON: {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown: {path_to_str(md_path)}")

    log("STEP", "Final summary.")
    log("OK", f"Cities inspected: {len(city_reports)}")
    log("OK", f"Files inspected: {len(all_file_reports)}")
    log("OK", f"Cities usable for finalization: {n_usable}")
    log("OK", f"Cities not usable yet: {n_not_usable}")

    log("INFO", "Finalization route counts:")
    for route, count in sorted(route_counts.items()):
        log("INFO", f"  {route}: {count}")

    if n_not_usable > 0:
        log("WARN", "Some cities are not usable yet. Inspect rtc_georeferencing.md.")
        if args.fail_if_unusable:
            raise SystemExit(2)


if __name__ == "__main__":
    main()