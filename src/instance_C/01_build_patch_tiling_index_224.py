#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_build_patch_tiling_index_224.py

Build the 224x224 patch tiling index for Instance C.

This script validates and indexes the aligned Instance C raster stack:

    S2 filled:
        <instance-root>/s2_filled/<city>/<city>_s2_*.tif
        Expected bands: 12

    S1 SNAP-GRD:
        <instance-root>/s1_ready/<city>/<city>_s1_*.tif
        Expected bands: 3
        Convention: VV, VH, VV_minus_VH

    S1 RTC:
        <instance-root>/s1_rtc_ready/<city>/<city>_s1_rtc_vv_vh_10m_aligned.tif
        Expected bands: 2
        Convention: VV, VH

    Labels:
        <instance-root>/labels/<city>/<city>_label_final.tif
        Expected bands: 1

Important:
    RTC is intentionally 2 bands only.
    Do NOT expect VV_minus_VH for RTC.
    This is required for CROMA-compatible SAR input [2, 224, 224].

Outputs:

    <instance-root>/metadata/instance_C_patches/
        patch_tiling_index_ps224_st112_cover.csv
        patch_tiling_index_ps224_st112_cover.json
        patch_tiling_index_ps224_st112_cover.md

Example:

python src/instance_C/01_build_patch_tiling_index_224.py `
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
# CSV / JSON / Markdown I/O
# ---------------------------------------------------------------------

def read_csv_rows_optional(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: Path, rows: List[Dict[str, object]], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    if not rows:
        fail(f"No rows to write: {path_to_str(path)}")

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

    lines.append("# Instance C patch tiling index")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- S2 root: `{summary['s2_root']}`")
    lines.append(f"- S1 SNAP-GRD root: `{summary['s1_snap_root']}`")
    lines.append(f"- S1 RTC root: `{summary['s1_rtc_root']}`")
    lines.append(f"- Label root: `{summary['label_root']}`")
    lines.append(f"- Output CSV: `{summary['outputs']['csv']}`")
    lines.append(f"- Patch size: `{summary['parameters']['patch_size']}`")
    lines.append(f"- Stride: `{summary['parameters']['stride']}`")
    lines.append(f"- Edge mode: `{summary['parameters']['edge_mode']}`")
    lines.append(f"- Cities indexed: `{summary['n_cities_indexed']}`")
    lines.append(f"- Total patches: `{summary['total_patches']}`")
    lines.append(f"- Validation failures: `{summary['n_validation_failures']}`")
    lines.append("")

    lines.append("## Band contract")
    lines.append("")
    lines.append("| Modality | Expected bands | Convention |")
    lines.append("|---|---:|---|")
    lines.append("| S2 | 12 | Sentinel-2 reflectance bands |")
    lines.append("| S1 SNAP-GRD | 3 | VV, VH, VV_minus_VH |")
    lines.append("| S1 RTC | 2 | VV, VH |")
    lines.append("| Label | 1 | Binary favela label |")
    lines.append("")

    lines.append("## City-level index summary")
    lines.append("")
    lines.append(
        "| city | region | patches | width | height | S2 bands | SNAP bands | RTC bands | label bands | status | notes |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")

    for row in city_rows:
        lines.append(
            f"| {row['city']} | "
            f"{row['region']} | "
            f"{row['n_patches']} | "
            f"{row['width']} | "
            f"{row['height']} | "
            f"{row['s2_band_count']} | "
            f"{row['s1_snap_band_count']} | "
            f"{row['s1_rtc_band_count']} | "
            f"{row['label_band_count']} | "
            f"{row['status']} | "
            f"{row['notes']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- This index intentionally accepts RTC as a 2-band VV/VH product.")
    lines.append("- SNAP-GRD remains a 3-band product because it includes VV_minus_VH.")
    lines.append("- For the main CROMA RTC-vs-SNAP comparison, both radar variants should use VV/VH only.")
    lines.append("- The patch index stores source paths and band counts so downstream metadata and dataloaders can choose the correct modality.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# City-region table
# ---------------------------------------------------------------------

def load_city_region_map(path: Path) -> Dict[str, str]:
    rows = read_csv_rows_optional(path)

    if not rows:
        log("WARN", f"City-region table not found or empty: {path_to_str(path)}")
        return {}

    fieldnames = set(rows[0].keys())

    city_col_candidates = ["city", "city_name", "name"]
    region_col_candidates = ["region", "macroregion", "macro_region"]

    city_col = None
    region_col = None

    for col in city_col_candidates:
        if col in fieldnames:
            city_col = col
            break

    for col in region_col_candidates:
        if col in fieldnames:
            region_col = col
            break

    if city_col is None or region_col is None:
        log(
            "WARN",
            f"Could not infer city/region columns from {path_to_str(path)}. "
            f"Columns are: {sorted(fieldnames)}"
        )
        return {}

    mapping: Dict[str, str] = {}

    for row in rows:
        city = normalize_city(row[city_col])
        region = str(row[region_col]).strip()
        mapping[city] = region

    return mapping


def default_city_region_table(instance_root: Path) -> Path:
    """
    Typical structure:

        D:/post_processing_dataset/dataset_instances/instance_C.../
        D:/post_processing_dataset/metadata/city_region_table.csv

    This function walks upward and checks common locations.
    """

    candidates = [
        instance_root.parent.parent / "metadata" / "city_region_table.csv",
        instance_root.parent / "metadata" / "city_region_table.csv",
        instance_root / "metadata" / "city_region_table.csv",
        Path("D:/post_processing_dataset/metadata/city_region_table.csv"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


# ---------------------------------------------------------------------
# Raster discovery
# ---------------------------------------------------------------------

EXCLUDE_NAME_PARTS = (
    "valid_mask",
    "fill_level",
    "fill_source",
    "nodata",
    "qa",
    "mask_before",
    "mask_after",
    "summary",
    "preview",
    "png",
    "jpg",
)


def is_excluded_raster(path: Path) -> bool:
    lower = path.name.lower()
    return any(part in lower for part in EXCLUDE_NAME_PARTS)


def candidate_tifs(folder: Path, patterns: Sequence[str]) -> List[Path]:
    matches: List[Path] = []

    for pattern in patterns:
        for path in folder.glob(pattern):
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
                if not is_excluded_raster(path):
                    matches.append(path)

    return sorted(set(matches))


def choose_single_raster(folder: Path, city: str, patterns: Sequence[str], label: str) -> Path:
    if not folder.exists():
        fail(f"Missing {label} folder for {city}: {path_to_str(folder)}")

    for pattern in patterns:
        matches = candidate_tifs(folder, [pattern])

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            formatted = "\n".join(f"  - {path_to_str(p)}" for p in matches[:30])
            fail(
                f"Ambiguous {label} raster for city {city} using pattern `{pattern}`:\n"
                f"{formatted}"
            )

    fail(f"Could not find {label} raster for city {city} in {path_to_str(folder)}")


def find_s2_path(s2_root: Path, city: str) -> Path:
    city = normalize_city(city)
    folder = s2_root / city

    patterns = [
        f"{city}_s2_12bands_reflectance_10m.tif",
        f"{city}_s2_12bands_reflectance_10m_filled.tif",
        f"{city}_s2_filled_12bands_reflectance_10m.tif",
        f"{city}*s2*12*reflectance*10m*.tif",
        f"{city}*s2*filled*.tif",
        "*s2*12*reflectance*10m*.tif",
        "*s2*filled*.tif",
        "*.tif",
        "*.tiff",
    ]

    return choose_single_raster(folder, city, patterns, "S2")


def find_s1_snap_path(s1_root: Path, city: str) -> Path:
    city = normalize_city(city)
    folder = s1_root / city

    patterns = [
        f"{city}_s1_snap_vv_vh_vvdiff_10m_aligned.tif",
        f"{city}_s1_grd_vv_vh_vvdiff_10m_aligned.tif",
        f"{city}*snap*vv*vh*.tif",
        f"{city}*grd*vv*vh*.tif",
        f"{city}*s1*vv*vh*vvdiff*.tif",
        f"{city}*s1*.tif",
        "*snap*vv*vh*.tif",
        "*grd*vv*vh*.tif",
        "*s1*vv*vh*vvdiff*.tif",
        "*s1*.tif",
        "*.tif",
        "*.tiff",
    ]

    return choose_single_raster(folder, city, patterns, "S1 SNAP-GRD")


def find_s1_rtc_path(s1_rtc_root: Path, city: str) -> Path:
    city = normalize_city(city)
    folder = s1_rtc_root / city

    patterns = [
        f"{city}_s1_rtc_vv_vh_10m_aligned.tif",
        f"{city}_s1_rtc_vv_vh_10m_aligned.tiff",
        f"{city}*s1_rtc*vv*vh*10m*aligned*.tif",
        f"{city}*rtc*vv*vh*.tif",
        "*s1_rtc*vv*vh*10m*aligned*.tif",
        "*rtc*vv*vh*.tif",
        "*.tif",
        "*.tiff",
    ]

    return choose_single_raster(folder, city, patterns, "S1 RTC")


def find_label_path(label_root: Path, city: str) -> Path:
    city = normalize_city(city)
    folder = label_root / city

    patterns = [
        f"{city}_label_final.tif",
        f"{city}_label_final.tiff",
        f"{city}*label_final*.tif",
        f"{city}*label*.tif",
        "*label_final*.tif",
        "*label*.tif",
        "*.tif",
        "*.tiff",
    ]

    return choose_single_raster(folder, city, patterns, "label")


# ---------------------------------------------------------------------
# Raster validation
# ---------------------------------------------------------------------

def affine_six(transform) -> Tuple[float, float, float, float, float, float]:
    return (
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    )


def transforms_equal(a, b, tolerance: float) -> bool:
    aa = affine_six(a)
    bb = affine_six(b)

    return all(abs(x - y) <= tolerance for x, y in zip(aa, bb))


def raster_info(path: Path) -> Dict[str, object]:
    with rasterio.open(path) as src:
        return {
            "path": path_to_str(path),
            "band_count": int(src.count),
            "width": int(src.width),
            "height": int(src.height),
            "crs": "" if src.crs is None else str(src.crs),
            "transform": affine_six(src.transform),
            "dtype": ";".join(str(x) for x in src.dtypes),
            "descriptions": ";".join("" if d is None else str(d) for d in src.descriptions),
        }


def validate_stack_for_city(
    *,
    city: str,
    s2_path: Path,
    s1_snap_path: Path,
    s1_rtc_path: Path,
    label_path_: Path,
    expected_s2_bands: int,
    expected_s1_snap_bands: int,
    expected_s1_rtc_bands: int,
    expected_label_bands: int,
    transform_tolerance: float,
) -> Tuple[Dict[str, object], List[str]]:
    errors: List[str] = []

    with rasterio.open(s2_path) as s2, \
         rasterio.open(s1_snap_path) as snap, \
         rasterio.open(s1_rtc_path) as rtc, \
         rasterio.open(label_path_) as label:

        if s2.count != expected_s2_bands:
            errors.append(f"S2 band count = {s2.count}, expected {expected_s2_bands}")

        if snap.count != expected_s1_snap_bands:
            errors.append(f"S1_SNAP band count = {snap.count}, expected {expected_s1_snap_bands}")

        if rtc.count != expected_s1_rtc_bands:
            errors.append(f"S1_RTC band count = {rtc.count}, expected {expected_s1_rtc_bands}")

        if label.count != expected_label_bands:
            errors.append(f"Label band count = {label.count}, expected {expected_label_bands}")

        reference = {
            "width": s2.width,
            "height": s2.height,
            "crs": s2.crs,
            "transform": s2.transform,
        }

        for name, src in [
            ("S1_SNAP", snap),
            ("S1_RTC", rtc),
            ("Label", label),
        ]:
            if src.width != reference["width"] or src.height != reference["height"]:
                errors.append(
                    f"{name} shape mismatch: "
                    f"{src.width}x{src.height} != {reference['width']}x{reference['height']}"
                )

            if src.crs != reference["crs"]:
                errors.append(f"{name} CRS mismatch: {src.crs} != {reference['crs']}")

            if not transforms_equal(src.transform, reference["transform"], transform_tolerance):
                errors.append(f"{name} transform mismatch with S2")

        info = {
            "width": int(s2.width),
            "height": int(s2.height),
            "crs": "" if s2.crs is None else str(s2.crs),
            "transform": affine_six(s2.transform),
            "s2_band_count": int(s2.count),
            "s1_snap_band_count": int(snap.count),
            "s1_rtc_band_count": int(rtc.count),
            "label_band_count": int(label.count),
        }

    return info, errors


# ---------------------------------------------------------------------
# Patch grid
# ---------------------------------------------------------------------

def build_axis_starts(length: int, patch_size: int, stride: int, edge_mode: str) -> List[int]:
    if length < patch_size:
        fail(f"Raster dimension {length} is smaller than patch size {patch_size}")

    starts = list(range(0, length - patch_size + 1, stride))

    if not starts:
        starts = [0]

    if edge_mode == "cover":
        last = length - patch_size
        if starts[-1] != last:
            starts.append(last)

    elif edge_mode == "drop":
        pass

    else:
        fail(f"Unsupported edge mode: {edge_mode}")

    return sorted(set(starts))


def build_patch_rows_for_city(
    *,
    city: str,
    region: str,
    width: int,
    height: int,
    patch_size: int,
    stride: int,
    edge_mode: str,
    s2_path: Path,
    s1_snap_path: Path,
    s1_rtc_path: Path,
    label_path_: Path,
    s2_band_count: int,
    s1_snap_band_count: int,
    s1_rtc_band_count: int,
    label_band_count: int,
) -> List[Dict[str, object]]:
    row_starts = build_axis_starts(height, patch_size, stride, edge_mode)
    col_starts = build_axis_starts(width, patch_size, stride, edge_mode)

    rows: List[Dict[str, object]] = []

    city = normalize_city(city)

    patch_counter = 0

    for row_start in row_starts:
        for col_start in col_starts:
            patch_counter += 1

            patch_id = (
                f"{city}"
                f"__r{int(row_start):06d}"
                f"__c{int(col_start):06d}"
                f"__ps{int(patch_size)}"
                f"__st{int(stride)}"
            )

            rows.append(
                {
                    "patch_id": patch_id,
                    "city": city,
                    "region": region,
                    "row_start": int(row_start),
                    "col_start": int(col_start),
                    "height": int(patch_size),
                    "width": int(patch_size),
                    "patch_size": int(patch_size),
                    "stride": int(stride),
                    "edge_mode": edge_mode,
                    "city_width": int(width),
                    "city_height": int(height),
                    "source_s2_path": path_to_str(s2_path),
                    "source_s1_path": path_to_str(s1_snap_path),
                    "source_s1_snap_path": path_to_str(s1_snap_path),
                    "source_s1_rtc_path": path_to_str(s1_rtc_path),
                    "source_label_path": path_to_str(label_path_),
                    "s2_exists": True,
                    "s1_exists": True,
                    "s1_snap_exists": True,
                    "s1_rtc_exists": True,
                    "label_exists": True,
                    "s2_band_count": int(s2_band_count),
                    "s1_band_count": int(s1_snap_band_count),
                    "s1_snap_band_count": int(s1_snap_band_count),
                    "s1_rtc_band_count": int(s1_rtc_band_count),
                    "label_band_count": int(label_band_count),
                }
            )

    return rows


# ---------------------------------------------------------------------
# City discovery
# ---------------------------------------------------------------------

def discover_cities(s2_root: Path) -> List[str]:
    if not s2_root.exists():
        fail(f"S2 root does not exist: {path_to_str(s2_root)}")

    cities = [
        normalize_city(p.name)
        for p in s2_root.iterdir()
        if p.is_dir()
    ]

    cities = sorted(set(cities))

    if not cities:
        fail(f"No city folders found in S2 root: {path_to_str(s2_root)}")

    return cities


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    *,
    instance_root: Path,
    s2_root: Path,
    s1_snap_root: Path,
    s1_rtc_root: Path,
    label_root: Path,
    city_region_table: Path,
    patch_rows: List[Dict[str, object]],
    city_rows: List[Dict[str, object]],
    validation_failures: List[str],
    args: argparse.Namespace,
    csv_path: Path,
    json_path: Path,
    md_path: Path,
) -> Dict[str, object]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "s2_root": path_to_str(s2_root),
        "s1_snap_root": path_to_str(s1_snap_root),
        "s1_rtc_root": path_to_str(s1_rtc_root),
        "label_root": path_to_str(label_root),
        "city_region_table": path_to_str(city_region_table),
        "n_cities_indexed": len(city_rows),
        "total_patches": len(patch_rows),
        "n_validation_failures": len(validation_failures),
        "validation_failures": validation_failures,
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "expected_s2_bands": args.expected_s2_bands,
            "expected_s1_snap_bands": args.expected_s1_snap_bands,
            "expected_s1_rtc_bands": args.expected_s1_rtc_bands,
            "expected_label_bands": args.expected_label_bands,
            "transform_tolerance": args.transform_tolerance,
        },
        "outputs": {
            "csv": path_to_str(csv_path),
            "json": path_to_str(json_path),
            "markdown": path_to_str(md_path),
        },
        "city_rows": city_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Instance C 224x224 patch tiling index with 2-band RTC support."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--s2-root",
        type=Path,
        default=None,
        help="Default: <instance-root>/s2_filled.",
    )

    parser.add_argument(
        "--s1-root",
        type=Path,
        default=None,
        help="S1 SNAP-GRD root. Default: <instance-root>/s1_ready.",
    )

    parser.add_argument(
        "--s1-rtc-root",
        type=Path,
        default=None,
        help="S1 RTC root. Default: <instance-root>/s1_rtc_ready.",
    )

    parser.add_argument(
        "--label-root",
        type=Path,
        default=None,
        help="Default: <instance-root>/labels.",
    )

    parser.add_argument(
        "--city-region-table",
        type=Path,
        default=None,
        help="Default: auto-detect city_region_table.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/instance_C_patches.",
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
        help="Patch stride. Default: 112.",
    )

    parser.add_argument(
        "--edge-mode",
        choices=["cover", "drop"],
        default="cover",
        help="Use cover to include edge patches. Default: cover.",
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
        "--transform-tolerance",
        type=float,
        default=0.0,
        help="Affine transform tolerance. Default: 0.0 exact match.",
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Optional city subset.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root

    s2_root: Path = args.s2_root or (instance_root / "s2_filled")
    s1_snap_root: Path = args.s1_root or (instance_root / "s1_ready")
    s1_rtc_root: Path = args.s1_rtc_root or (instance_root / "s1_rtc_ready")
    label_root: Path = args.label_root or (instance_root / "labels")
    output_dir: Path = args.output_dir or (instance_root / "metadata" / "instance_C_patches")
    city_region_table: Path = args.city_region_table or default_city_region_table(instance_root)

    output_stem = f"patch_tiling_index_ps{args.patch_size}_st{args.stride}_{args.edge_mode}"
    csv_path = output_dir / f"{output_stem}.csv"
    json_path = output_dir / f"{output_stem}.json"
    md_path = output_dir / f"{output_stem}.md"

    log("STEP", "Building instance C 224x224 patch tiling index.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"S2 root:       {path_to_str(s2_root)}")
    log("INFO", f"S1 root:       {path_to_str(s1_snap_root)}")
    log("INFO", f"Label root:    {path_to_str(label_root)}")
    log("INFO", f"S1 RTC root:   {path_to_str(s1_rtc_root)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")
    log("INFO", f"Loading city-region table: {path_to_str(city_region_table)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    city_region_map = load_city_region_map(city_region_table)

    if args.cities:
        cities = sorted(normalize_city(c) for c in args.cities)
    else:
        cities = discover_cities(s2_root)

    log("INFO", f"Cities to index: {len(cities)}")

    all_patch_rows: List[Dict[str, object]] = []
    city_summary_rows: List[Dict[str, object]] = []
    validation_failures: List[str] = []

    for city in cities:
        city = normalize_city(city)
        region = city_region_map.get(city, "unknown")

        log("STEP", f"Processing city: {city}")

        try:
            s2_path = find_s2_path(s2_root, city)
            s1_snap_path = find_s1_snap_path(s1_snap_root, city)
            s1_rtc_path = find_s1_rtc_path(s1_rtc_root, city)
            label_path_ = find_label_path(label_root, city)

            info, errors = validate_stack_for_city(
                city=city,
                s2_path=s2_path,
                s1_snap_path=s1_snap_path,
                s1_rtc_path=s1_rtc_path,
                label_path_=label_path_,
                expected_s2_bands=int(args.expected_s2_bands),
                expected_s1_snap_bands=int(args.expected_s1_snap_bands),
                expected_s1_rtc_bands=int(args.expected_s1_rtc_bands),
                expected_label_bands=int(args.expected_label_bands),
                transform_tolerance=float(args.transform_tolerance),
            )

            if errors:
                for err in errors:
                    message = f"{city}: {err}"
                    validation_failures.append(message)
                    log("ERROR", message)

                city_summary_rows.append(
                    {
                        "city": city,
                        "region": region,
                        "status": "failed",
                        "n_patches": 0,
                        "width": info.get("width", ""),
                        "height": info.get("height", ""),
                        "s2_band_count": info.get("s2_band_count", ""),
                        "s1_snap_band_count": info.get("s1_snap_band_count", ""),
                        "s1_rtc_band_count": info.get("s1_rtc_band_count", ""),
                        "label_band_count": info.get("label_band_count", ""),
                        "s2_path": path_to_str(s2_path),
                        "s1_snap_path": path_to_str(s1_snap_path),
                        "s1_rtc_path": path_to_str(s1_rtc_path),
                        "label_path": path_to_str(label_path_),
                        "notes": " | ".join(errors),
                    }
                )
                continue

            patch_rows = build_patch_rows_for_city(
                city=city,
                region=region,
                width=int(info["width"]),
                height=int(info["height"]),
                patch_size=int(args.patch_size),
                stride=int(args.stride),
                edge_mode=str(args.edge_mode),
                s2_path=s2_path,
                s1_snap_path=s1_snap_path,
                s1_rtc_path=s1_rtc_path,
                label_path_=label_path_,
                s2_band_count=int(info["s2_band_count"]),
                s1_snap_band_count=int(info["s1_snap_band_count"]),
                s1_rtc_band_count=int(info["s1_rtc_band_count"]),
                label_band_count=int(info["label_band_count"]),
            )

            all_patch_rows.extend(patch_rows)

            city_summary_rows.append(
                {
                    "city": city,
                    "region": region,
                    "status": "ok",
                    "n_patches": len(patch_rows),
                    "width": info["width"],
                    "height": info["height"],
                    "s2_band_count": info["s2_band_count"],
                    "s1_snap_band_count": info["s1_snap_band_count"],
                    "s1_rtc_band_count": info["s1_rtc_band_count"],
                    "label_band_count": info["label_band_count"],
                    "s2_path": path_to_str(s2_path),
                    "s1_snap_path": path_to_str(s1_snap_path),
                    "s1_rtc_path": path_to_str(s1_rtc_path),
                    "label_path": path_to_str(label_path_),
                    "notes": "",
                }
            )

            log(
                "OK",
                f"{city}: patches={len(patch_rows)}, "
                f"size={info['width']}x{info['height']}, "
                f"S2={info['s2_band_count']} bands, "
                f"SNAP={info['s1_snap_band_count']} bands, "
                f"RTC={info['s1_rtc_band_count']} bands",
            )

        except Exception as exc:
            message = f"{city}: {repr(exc)}"
            validation_failures.append(message)
            log("ERROR", message)

            city_summary_rows.append(
                {
                    "city": city,
                    "region": region,
                    "status": "failed",
                    "n_patches": 0,
                    "width": "",
                    "height": "",
                    "s2_band_count": "",
                    "s1_snap_band_count": "",
                    "s1_rtc_band_count": "",
                    "label_band_count": "",
                    "s2_path": "",
                    "s1_snap_path": "",
                    "s1_rtc_path": "",
                    "label_path": "",
                    "notes": repr(exc),
                }
            )

    if validation_failures:
        log("ERROR", "Alignment or band-count validation failed for one or more cities:")
        for failure in validation_failures:
            log("ERROR", f"  - {failure}")
        raise SystemExit(2)

    if not all_patch_rows:
        fail("No patch rows were created.")

    summary = build_summary(
        instance_root=instance_root,
        s2_root=s2_root,
        s1_snap_root=s1_snap_root,
        s1_rtc_root=s1_rtc_root,
        label_root=label_root,
        city_region_table=city_region_table,
        patch_rows=all_patch_rows,
        city_rows=city_summary_rows,
        validation_failures=validation_failures,
        args=args,
        csv_path=csv_path,
        json_path=json_path,
        md_path=md_path,
    )

    log("STEP", "Writing outputs.")

    write_csv(csv_path, all_patch_rows, overwrite=bool(args.overwrite))
    write_json(json_path, summary, overwrite=bool(args.overwrite))
    write_markdown(md_path, summary, city_summary_rows, overwrite=bool(args.overwrite))

    log("OK", f"Wrote CSV:      {path_to_str(csv_path)}")
    log("OK", f"Wrote JSON:     {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown: {path_to_str(md_path)}")

    log("STEP", "Final summary.")
    log("OK", f"Cities indexed: {summary['n_cities_indexed']}")
    log("OK", f"Total patches: {summary['total_patches']}")
    log("OK", f"Validation failures: {summary['n_validation_failures']}")

    rtc_available = sum(1 for row in all_patch_rows if row["s1_rtc_exists"])
    log("OK", f"Patches with S1 RTC available: {rtc_available}")


if __name__ == "__main__":
    main()