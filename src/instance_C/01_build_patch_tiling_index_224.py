#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_build_patch_tiling_index_224.py

Build a 224x224 overlapping patch tiling index for the repaired instance C dataset.

This script DOES NOT export physical patch images.
It only creates a metadata index so downstream dataloaders can read GeoTIFF windows directly.

Expected instance structure:

instance_C_s2_nodata_repaired/
    s2_filled/<city>/<city>_s2_12bands_reflectance_10m.tif
    s1_ready/<city>/<city>_s1_ready_vv_vh_vvdiff_10m_aligned.tif
    labels/<city>/<city>_label_final.tif
    s1_rtc_ready/<city>/<city>_s1_rtc_vv_vh_vvdiff_10m_aligned.tif  # optional / future

Default output:

instance_C_s2_nodata_repaired/
    metadata/
        instance_C_patches/
            patch_tiling_index_ps224_st112_cover.csv
            patch_tiling_index_ps224_st112_cover.json
            patch_tiling_index_ps224_st112_cover.md

Example PowerShell command:

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
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import rasterio
except ImportError as exc:
    raise SystemExit(
        "[ERROR] rasterio is required but is not installed in this environment.\n"
        "Install it first, for example:\n"
        "    pip install rasterio\n\n"
        f"Original import error: {exc}"
    )


# ---------------------------------------------------------------------
# Fallback city-region mapping
# ---------------------------------------------------------------------

FALLBACK_CITY_TO_REGION: Dict[str, str] = {
    "belem": "North",
    "manaus": "North",

    "fortaleza": "Northeast",
    "joao_pessoa": "Northeast",
    "maceio": "Northeast",
    "natal": "Northeast",
    "recife": "Northeast",
    "salvador": "Northeast",
    "sao_luis": "Northeast",
    "teresina": "Northeast",

    "brasilia": "Central-West",
    "campo_grande": "Central-West",
    "goiania": "Central-West",

    "belo_horizonte": "Southeast",
    "campinas": "Southeast",
    "duque_de_caxias": "Southeast",
    "guarulhos": "Southeast",
    "nova_iguacu": "Southeast",
    "rio_de_janeiro": "Southeast",
    "santo_andre": "Southeast",
    "sao_bernardo_do_campo": "Southeast",
    "sao_goncalo": "Southeast",
    "sao_paulo": "Southeast",
    "sorocaba": "Southeast",

    "curitiba": "South",
    "porto_alegre": "South",
}


QA_NAME_PARTS = (
    "valid_mask",
    "fill_level",
    "fill_source",
    "nodata",
    "qa",
    "mask_before",
    "mask_after",
    "summary",
)


# ---------------------------------------------------------------------
# Logging and utility helpers
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


def normalize_city_name(value: str) -> str:
    return value.strip().replace("\\", "/").split("/")[-1]


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def is_probably_qa_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(part in lower_name for part in QA_NAME_PARTS)


def filter_non_qa_tifs(paths: Iterable[Path]) -> List[Path]:
    clean_paths = []

    for path in paths:
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue

        if is_probably_qa_file(path):
            continue

        clean_paths.append(path)

    return sorted(set(clean_paths))


# ---------------------------------------------------------------------
# City discovery and region metadata
# ---------------------------------------------------------------------

def discover_cities_from_s2(s2_root: Path) -> List[str]:
    """
    Discover cities from s2_filled/.

    Expected structure:
        s2_filled/<city>/<city>_s2_12bands_reflectance_10m.tif
    """

    if not s2_root.exists():
        fail(f"S2 root does not exist: {path_to_str(s2_root)}")

    city_dirs = [
        child.name
        for child in sorted(s2_root.iterdir())
        if child.is_dir()
    ]

    if not city_dirs:
        fail(
            "No city folders were found under s2_filled/:\n"
            f"  {path_to_str(s2_root)}"
        )

    return [normalize_city_name(city) for city in city_dirs]


def find_city_region_table(instance_root: Path, explicit_table: Optional[Path]) -> Optional[Path]:
    if explicit_table is not None:
        if explicit_table.exists():
            return explicit_table
        fail(f"Explicit --city-region-table does not exist: {path_to_str(explicit_table)}")

    candidates = [
        instance_root / "metadata" / "city_region_table.csv",
        instance_root / "city_region_table.csv",
        instance_root.parent / "metadata" / "city_region_table.csv",
        instance_root.parent.parent / "metadata" / "city_region_table.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_city_region_mapping(
    instance_root: Path,
    explicit_table: Optional[Path],
) -> Tuple[Dict[str, str], Optional[Path]]:
    """
    Load city-region mapping.

    If a city_region_table.csv exists, use it.
    Otherwise, use the built-in 26-city fallback mapping.
    """

    table_path = find_city_region_table(instance_root, explicit_table)

    mapping: Dict[str, str] = {}

    if table_path is None:
        log(
            "WARN",
            "No city_region_table.csv found. Using built-in fallback city-region mapping.",
        )
        mapping.update(FALLBACK_CITY_TO_REGION)
        return mapping, None

    log("INFO", f"Loading city-region table: {path_to_str(table_path)}")

    with table_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            fail(f"City-region table has no header: {path_to_str(table_path)}")

        lower_to_original = {
            col.lower().strip(): col
            for col in reader.fieldnames
        }

        city_col = None
        for candidate in ("city", "city_slug", "city_name", "name"):
            if candidate in lower_to_original:
                city_col = lower_to_original[candidate]
                break

        region_col = None
        for candidate in ("region", "macroregion", "brazil_region", "geographic_region"):
            if candidate in lower_to_original:
                region_col = lower_to_original[candidate]
                break

        if city_col is None or region_col is None:
            fail(
                "Could not identify city and region columns in city-region table.\n"
                f"Table: {path_to_str(table_path)}\n"
                f"Columns found: {reader.fieldnames}\n"
                "Expected a city column like 'city' or 'city_slug', and a region column like 'region'."
            )

        for row in reader:
            city = normalize_city_name(str(row.get(city_col, "")))
            region = str(row.get(region_col, "")).strip()

            if city and region:
                mapping[city] = region

    # Supplement any missing city from the fallback mapping without overwriting the CSV.
    for city, region in FALLBACK_CITY_TO_REGION.items():
        mapping.setdefault(city, region)

    return mapping, table_path


# ---------------------------------------------------------------------
# Raster discovery
# ---------------------------------------------------------------------

def find_one_raster(
    folder: Path,
    patterns: Sequence[str],
    label: str,
    allow_missing: bool = False,
) -> Optional[Path]:
    """
    Find exactly one non-QA raster using ordered glob patterns.

    The strictest patterns should be first.
    If a broad pattern matches multiple files, we fail rather than choosing silently.
    """

    if not folder.exists():
        if allow_missing:
            return None
        fail(f"Missing {label} folder: {path_to_str(folder)}")

    for pattern in patterns:
        matches = filter_non_qa_tifs(folder.glob(pattern))

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            formatted = "\n".join(f"    - {path_to_str(p)}" for p in matches[:25])
            if len(matches) > 25:
                formatted += f"\n    ... and {len(matches) - 25} more"

            fail(
                f"Ambiguous {label} raster in:\n"
                f"  {path_to_str(folder)}\n"
                f"Pattern:\n"
                f"  {pattern}\n"
                f"Matched files:\n"
                f"{formatted}\n\n"
                "Please clean/rename the folder or make the filename more explicit."
            )

    if allow_missing:
        return None

    fail(
        f"Could not find {label} raster in:\n"
        f"  {path_to_str(folder)}\n"
        "Patterns tried:\n"
        + "\n".join(f"  - {pattern}" for pattern in patterns)
    )


def find_city_files(instance_root: Path, city: str) -> Dict[str, Optional[Path]]:
    s2_dir = instance_root / "s2_filled" / city
    s1_dir = instance_root / "s1_ready" / city
    label_dir = instance_root / "labels" / city
    rtc_dir = instance_root / "s1_rtc_ready" / city

    s2_path = find_one_raster(
        s2_dir,
        label="S2",
        patterns=[
            f"{city}_s2_12bands_reflectance_10m.tif",
            f"{city}_s2_12bands_reflectance_10m.tiff",
            f"{city}_s2_12bands_reflectance_10m_filled.tif",
            f"{city}_s2_filled_12bands_reflectance_10m.tif",
            f"{city}*s2*12*reflectance*10m*.tif",
            f"{city}*s2*filled*.tif",
            "*s2*12*reflectance*10m*.tif",
            "*s2*filled*.tif",
            "*.tif",
            "*.tiff",
        ],
    )

    s1_snap_grd_path = find_one_raster(
        s1_dir,
        label="S1_SNAP_GRD",
        patterns=[
            f"{city}_s1_ready_vv_vh_vvdiff_10m_aligned.tif",
            f"{city}_s1_ready_vv_vh_vvdiff_10m_aligned.tiff",
            f"{city}*s1_ready*vv*vh*vvdiff*10m*aligned*.tif",
            f"{city}*s1*ready*.tif",
            "*s1_ready*vv*vh*vvdiff*10m*aligned*.tif",
            "*s1*ready*.tif",
            "*.tif",
            "*.tiff",
        ],
    )

    label_path = find_one_raster(
        label_dir,
        label="label",
        patterns=[
            f"{city}_label_final.tif",
            f"{city}_label_final.tiff",
            f"{city}*label_final*.tif",
            f"{city}*label*.tif",
            "*label_final*.tif",
            "*label*.tif",
            "*.tif",
            "*.tiff",
        ],
    )

    s1_rtc_path = find_one_raster(
        rtc_dir,
        label="S1_RTC",
        allow_missing=True,
        patterns=[
            f"{city}_s1_rtc_vv_vh_vvdiff_10m_aligned.tif",
            f"{city}_s1_rtc_vv_vh_vvdiff_10m_aligned.tiff",
            f"{city}*s1_rtc*vv*vh*vvdiff*10m*aligned*.tif",
            f"{city}*rtc*.tif",
            "*s1_rtc*vv*vh*vvdiff*10m*aligned*.tif",
            "*rtc*.tif",
            "*.tif",
            "*.tiff",
        ],
    )

    return {
        "s2": s2_path,
        "s1_snap_grd": s1_snap_grd_path,
        "label": label_path,
        "s1_rtc": s1_rtc_path,
    }


# ---------------------------------------------------------------------
# Raster validation
# ---------------------------------------------------------------------

def transforms_equal(a, b, tolerance: float) -> bool:
    if tolerance <= 0:
        return a == b

    return all(
        abs(float(x) - float(y)) <= tolerance
        for x, y in zip(tuple(a), tuple(b))
    )


def validate_city_stack(
    city: str,
    files: Dict[str, Optional[Path]],
    expected_s2_bands: int,
    expected_s1_bands: int,
    expected_label_bands: int,
    transform_tolerance: float,
    require_s1_rtc: bool,
) -> Dict[str, object]:
    s2_path = files["s2"]
    s1_path = files["s1_snap_grd"]
    label_path = files["label"]
    rtc_path = files["s1_rtc"]

    if s2_path is None:
        fail(f"{city}: missing S2 path.")
    if s1_path is None:
        fail(f"{city}: missing S1_SNAP_GRD path.")
    if label_path is None:
        fail(f"{city}: missing label path.")

    problems: List[str] = []

    with rasterio.open(s2_path) as s2, rasterio.open(s1_path) as s1, rasterio.open(label_path) as label:
        if s2.count != expected_s2_bands:
            problems.append(f"S2 band count = {s2.count}, expected {expected_s2_bands}")

        if s1.count != expected_s1_bands:
            problems.append(f"S1_SNAP_GRD band count = {s1.count}, expected {expected_s1_bands}")

        if label.count != expected_label_bands:
            problems.append(f"label band count = {label.count}, expected {expected_label_bands}")

        if s1.width != s2.width or s1.height != s2.height:
            problems.append(
                f"S1_SNAP_GRD shape ({s1.height}, {s1.width}) != "
                f"S2 shape ({s2.height}, {s2.width})"
            )

        if label.width != s2.width or label.height != s2.height:
            problems.append(
                f"label shape ({label.height}, {label.width}) != "
                f"S2 shape ({s2.height}, {s2.width})"
            )

        if s1.crs != s2.crs:
            problems.append(f"S1_SNAP_GRD CRS {s1.crs} != S2 CRS {s2.crs}")

        if label.crs != s2.crs:
            problems.append(f"label CRS {label.crs} != S2 CRS {s2.crs}")

        if not transforms_equal(s1.transform, s2.transform, transform_tolerance):
            problems.append("S1_SNAP_GRD transform does not match S2 transform")

        if not transforms_equal(label.transform, s2.transform, transform_tolerance):
            problems.append("label transform does not match S2 transform")

        raster_height = s2.height
        raster_width = s2.width
        crs = str(s2.crs)
        transform = tuple(float(v) for v in s2.transform)

    rtc_status = "missing"

    if rtc_path is not None and rtc_path.exists():
        rtc_status = "present"

        with rasterio.open(s2_path) as s2, rasterio.open(rtc_path) as rtc:
            if rtc.count != expected_s1_bands:
                problems.append(f"S1_RTC band count = {rtc.count}, expected {expected_s1_bands}")

            if rtc.width != s2.width or rtc.height != s2.height:
                problems.append(
                    f"S1_RTC shape ({rtc.height}, {rtc.width}) != "
                    f"S2 shape ({s2.height}, {s2.width})"
                )

            if rtc.crs != s2.crs:
                problems.append(f"S1_RTC CRS {rtc.crs} != S2 CRS {s2.crs}")

            if not transforms_equal(rtc.transform, s2.transform, transform_tolerance):
                problems.append("S1_RTC transform does not match S2 transform")

    elif require_s1_rtc:
        problems.append("S1_RTC is required but missing")

    return {
        "ok": len(problems) == 0,
        "problems": problems,
        "raster_height": raster_height,
        "raster_width": raster_width,
        "crs": crs,
        "transform": transform,
        "rtc_status": rtc_status,
    }


# ---------------------------------------------------------------------
# Tiling logic
# ---------------------------------------------------------------------

def build_window_starts(size: int, patch_size: int, stride: int, edge_mode: str) -> List[int]:
    """
    Build start indices for one raster dimension.

    cover mode:
        Include final edge window at size - patch_size when needed.

    drop mode:
        Only include regular stride windows.
    """

    if patch_size <= 0:
        fail("--patch-size must be positive.")

    if stride <= 0:
        fail("--stride must be positive.")

    if size < patch_size:
        fail(
            f"Raster dimension {size} is smaller than patch_size={patch_size}. "
            "This script does not pad smaller rasters."
        )

    if edge_mode not in {"cover", "drop"}:
        fail(f"Unsupported edge_mode: {edge_mode}. Expected 'cover' or 'drop'.")

    starts = list(range(0, size - patch_size + 1, stride))

    if edge_mode == "cover":
        last_start = size - patch_size

        if not starts:
            starts = [last_start]
        elif starts[-1] != last_start:
            starts.append(last_start)

    # Remove duplicates while preserving sorted order.
    return sorted(set(starts))


def generate_city_patch_rows(
    city: str,
    region: str,
    raster_height: int,
    raster_width: int,
    patch_size: int,
    stride: int,
    edge_mode: str,
    files: Dict[str, Optional[Path]],
) -> Tuple[List[Dict[str, object]], int, int]:
    row_starts = build_window_starts(raster_height, patch_size, stride, edge_mode)
    col_starts = build_window_starts(raster_width, patch_size, stride, edge_mode)

    rows: List[Dict[str, object]] = []
    city_patch_index = 0

    for row_start in row_starts:
        for col_start in col_starts:
            patch_id = (
                f"{city}__ps{patch_size}_st{stride}"
                f"__r{row_start:06d}_c{col_start:06d}"
            )

            rows.append(
                {
                    "patch_id": patch_id,
                    "city": city,
                    "region": region,
                    "city_patch_index": city_patch_index,
                    "row_start": row_start,
                    "col_start": col_start,
                    "height": patch_size,
                    "width": patch_size,
                    "patch_size": patch_size,
                    "stride": stride,
                    "edge_mode": edge_mode,
                    "raster_height": raster_height,
                    "raster_width": raster_width,
                    "source_s2_path": path_to_str(files["s2"]),
                    "source_s1_snap_grd_path": path_to_str(files["s1_snap_grd"]),
                    "source_s1_rtc_path": path_to_str(files["s1_rtc"]),
                    "source_label_path": path_to_str(files["label"]),
                }
            )

            city_patch_index += 1

    return rows, len(row_starts), len(col_starts)


# ---------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, object]], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    if not rows:
        fail("No rows were generated. Refusing to write an empty CSV.")

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(path: Path, summary: Dict[str, object], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    city_summaries = summary["city_summaries"]
    patches_by_region = summary["patches_by_region"]

    lines: List[str] = []

    lines.append("# Instance C patch tiling index")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Patch size: `{summary['parameters']['patch_size']}`")
    lines.append(f"- Stride: `{summary['parameters']['stride']}`")
    lines.append(f"- Edge mode: `{summary['parameters']['edge_mode']}`")
    lines.append(f"- Cities indexed: `{summary['n_cities_indexed']}`")
    lines.append(f"- Total patches: `{summary['total_patches']}`")
    lines.append(f"- Missing required file count: `{summary['missing_required_file_count']}`")
    lines.append(f"- Alignment problem count: `{summary['alignment_problem_count']}`")
    lines.append(f"- RTC present cities: `{summary['rtc_present_city_count']}`")
    lines.append(f"- RTC missing cities: `{summary['rtc_missing_city_count']}`")
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- CSV: `{summary['outputs']['csv']}`")
    lines.append(f"- JSON: `{summary['outputs']['json']}`")
    lines.append(f"- Markdown: `{summary['outputs']['markdown']}`")
    lines.append("")

    lines.append("## Patch counts by city")
    lines.append("")
    lines.append(
        "| city | region | raster height | raster width | row windows | col windows | patches | RTC status |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---|"
    )

    for item in city_summaries:
        lines.append(
            f"| {item['city']} | {item['region']} | "
            f"{item['raster_height']} | {item['raster_width']} | "
            f"{item['n_row_windows']} | {item['n_col_windows']} | "
            f"{item['n_patches']} | {item['rtc_status']} |"
        )

    lines.append("")
    lines.append("## Patch counts by region")
    lines.append("")
    lines.append("| region | patches |")
    lines.append("|---|---:|")

    for region, count in sorted(patches_by_region.items()):
        lines.append(f"| {region} | {count} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This script writes only a patch index. It does not export physical patch rasters.")
    lines.append("- `source_s1_rtc_path` is blank when `s1_rtc_ready/` is not available yet.")
    lines.append("- Downstream CROMA/PyTorch dataloaders should read GeoTIFF windows using `row_start`, `col_start`, `height`, and `width`.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 224x224 overlapping patch tiling index for instance C."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size in pixels. Default: 224.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=112,
        help="Stride in pixels. Default: 112.",
    )

    parser.add_argument(
        "--edge-mode",
        choices=["cover", "drop"],
        default="cover",
        help="Window edge handling. 'cover' adds edge windows. Default: cover.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. "
            "Default: <instance-root>/metadata/instance_C_patches"
        ),
    )

    parser.add_argument(
        "--city-region-table",
        type=Path,
        default=None,
        help="Optional explicit path to city_region_table.csv.",
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Optional city subset for debugging. Default: all discovered cities.",
    )

    parser.add_argument(
        "--expected-city-count",
        type=int,
        default=26,
        help="Expected number of cities when --cities is not used. Default: 26.",
    )

    parser.add_argument(
        "--no-require-expected-city-count",
        action="store_true",
        help="Warn instead of failing if discovered city count differs from expected count.",
    )

    parser.add_argument(
        "--expected-s2-bands",
        type=int,
        default=12,
        help="Expected S2 band count. Default: 12.",
    )

    parser.add_argument(
        "--expected-s1-bands",
        type=int,
        default=3,
        help="Expected S1 band count. Default: 3.",
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
        help="Affine transform tolerance. Default 0.0 means exact match.",
    )

    parser.add_argument(
        "--require-s1-rtc",
        action="store_true",
        help="Fail if S1_RTC is missing. Use later when s1_rtc_ready/ exists.",
    )

    parser.add_argument(
        "--allow-unknown-region",
        action="store_true",
        help="Allow unknown city region and write region as UNKNOWN.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output CSV/JSON/Markdown files.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    instance_root: Path = args.instance_root
    output_dir: Path = args.output_dir or (
        instance_root / "metadata" / "instance_C_patches"
    )

    s2_root = instance_root / "s2_filled"
    s1_root = instance_root / "s1_ready"
    label_root = instance_root / "labels"
    rtc_root = instance_root / "s1_rtc_ready"

    log("STEP", "Building instance C 224x224 patch tiling index.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"S2 root:       {path_to_str(s2_root)}")
    log("INFO", f"S1 root:       {path_to_str(s1_root)}")
    log("INFO", f"Label root:    {path_to_str(label_root)}")
    log("INFO", f"S1 RTC root:   {path_to_str(rtc_root)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not s2_root.exists():
        fail(f"Missing required folder: {path_to_str(s2_root)}")

    if not s1_root.exists():
        fail(f"Missing required folder: {path_to_str(s1_root)}")

    if not label_root.exists():
        fail(f"Missing required folder: {path_to_str(label_root)}")

    if not rtc_root.exists():
        log(
            "WARN",
            "s1_rtc_ready/ does not exist yet. source_s1_rtc_path will be blank unless files are found later.",
        )

    city_region_mapping, city_region_table_used = load_city_region_mapping(
        instance_root=instance_root,
        explicit_table=args.city_region_table,
    )

    discovered_cities = discover_cities_from_s2(s2_root)

    if args.cities:
        requested_cities = [normalize_city_name(city) for city in args.cities]
        requested_set = set(requested_cities)

        missing_requested = sorted(requested_set - set(discovered_cities))
        if missing_requested:
            fail(
                "The following requested cities were not found under s2_filled/:\n"
                + "\n".join(f"  - {city}" for city in missing_requested)
            )

        cities = [city for city in discovered_cities if city in requested_set]
        log("WARN", f"Running on city subset: {', '.join(cities)}")
    else:
        cities = discovered_cities

    if not args.cities and len(cities) != args.expected_city_count:
        message = (
            f"Discovered {len(cities)} cities, but expected {args.expected_city_count}."
        )

        if args.no_require_expected_city_count:
            log("WARN", message)
        else:
            fail(
                message
                + "\nUse --no-require-expected-city-count only if this is intentional."
            )

    log("INFO", f"Cities to index: {len(cities)}")

    all_rows: List[Dict[str, object]] = []
    city_summaries: List[Dict[str, object]] = []
    alignment_problem_records: List[Dict[str, object]] = []

    for city in cities:
        log("STEP", f"Processing city: {city}")

        region = city_region_mapping.get(city)

        if not region:
            if args.allow_unknown_region:
                region = "UNKNOWN"
                log("WARN", f"{city}: region is unknown. Writing region as UNKNOWN.")
            else:
                fail(
                    f"No region found for city '{city}'. "
                    "Provide --city-region-table or use --allow-unknown-region."
                )

        files = find_city_files(instance_root, city)

        validation = validate_city_stack(
            city=city,
            files=files,
            expected_s2_bands=args.expected_s2_bands,
            expected_s1_bands=args.expected_s1_bands,
            expected_label_bands=args.expected_label_bands,
            transform_tolerance=args.transform_tolerance,
            require_s1_rtc=args.require_s1_rtc,
        )

        if not validation["ok"]:
            alignment_problem_records.append(
                {
                    "city": city,
                    "problems": validation["problems"],
                }
            )

            for problem in validation["problems"]:
                log("ERROR", f"{city}: {problem}")

            continue

        raster_height = int(validation["raster_height"])
        raster_width = int(validation["raster_width"])

        city_rows, n_row_windows, n_col_windows = generate_city_patch_rows(
            city=city,
            region=region,
            raster_height=raster_height,
            raster_width=raster_width,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            files=files,
        )

        all_rows.extend(city_rows)

        city_summary = {
            "city": city,
            "region": region,
            "raster_height": raster_height,
            "raster_width": raster_width,
            "n_row_windows": n_row_windows,
            "n_col_windows": n_col_windows,
            "n_patches": len(city_rows),
            "rtc_status": validation["rtc_status"],
            "crs": validation["crs"],
            "source_s2_path": path_to_str(files["s2"]),
            "source_s1_snap_grd_path": path_to_str(files["s1_snap_grd"]),
            "source_s1_rtc_path": path_to_str(files["s1_rtc"]),
            "source_label_path": path_to_str(files["label"]),
        }

        city_summaries.append(city_summary)

        log(
            "OK",
            f"{city}: {len(city_rows)} patches "
            f"({n_row_windows} row windows x {n_col_windows} col windows), "
            f"raster={raster_height}x{raster_width}, "
            f"region={region}, "
            f"RTC={validation['rtc_status']}",
        )

    if alignment_problem_records:
        formatted = []

        for record in alignment_problem_records:
            city = record["city"]
            problems = "; ".join(record["problems"])
            formatted.append(f"  - {city}: {problems}")

        fail(
            "Alignment or band-count validation failed for one or more cities:\n"
            + "\n".join(formatted)
        )

    if not all_rows:
        fail("No patch rows were generated.")

    patch_size = args.patch_size
    stride = args.stride
    edge_mode = args.edge_mode

    csv_path = output_dir / f"patch_tiling_index_ps{patch_size}_st{stride}_{edge_mode}.csv"
    json_path = output_dir / f"patch_tiling_index_ps{patch_size}_st{stride}_{edge_mode}.json"
    md_path = output_dir / f"patch_tiling_index_ps{patch_size}_st{stride}_{edge_mode}.md"

    patches_by_city = Counter(str(row["city"]) for row in all_rows)
    patches_by_region = Counter(str(row["region"]) for row in all_rows)

    rtc_present_city_count = sum(
        1 for item in city_summaries if item["rtc_status"] == "present"
    )
    rtc_missing_city_count = len(city_summaries) - rtc_present_city_count

    summary: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "source_roots": {
            "s2_filled": path_to_str(s2_root),
            "s1_ready": path_to_str(s1_root),
            "labels": path_to_str(label_root),
            "s1_rtc_ready": path_to_str(rtc_root),
        },
        "city_region_table_used": path_to_str(city_region_table_used),
        "parameters": {
            "patch_size": patch_size,
            "stride": stride,
            "edge_mode": edge_mode,
            "expected_city_count": args.expected_city_count,
            "expected_s2_bands": args.expected_s2_bands,
            "expected_s1_bands": args.expected_s1_bands,
            "expected_label_bands": args.expected_label_bands,
            "transform_tolerance": args.transform_tolerance,
            "require_s1_rtc": bool(args.require_s1_rtc),
        },
        "n_cities_indexed": len(city_summaries),
        "total_patches": len(all_rows),
        "missing_required_file_count": 0,
        "alignment_problem_count": len(alignment_problem_records),
        "rtc_present_city_count": rtc_present_city_count,
        "rtc_missing_city_count": rtc_missing_city_count,
        "patches_by_city": dict(sorted(patches_by_city.items())),
        "patches_by_region": dict(sorted(patches_by_region.items())),
        "city_summaries": city_summaries,
        "outputs": {
            "csv": path_to_str(csv_path),
            "json": path_to_str(json_path),
            "markdown": path_to_str(md_path),
        },
    }

    log("STEP", "Writing outputs.")

    write_csv(csv_path, all_rows, overwrite=args.overwrite)
    write_json(json_path, summary, overwrite=args.overwrite)
    write_markdown(md_path, summary, overwrite=args.overwrite)

    log("OK", f"Wrote CSV:      {path_to_str(csv_path)}")
    log("OK", f"Wrote JSON:     {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown: {path_to_str(md_path)}")

    log("STEP", "Final summary.")
    log("OK", f"Cities indexed: {len(city_summaries)}")
    log("OK", f"Total patches: {len(all_rows)}")
    log("OK", "Missing required file count: 0")
    log("OK", f"Alignment problem count: {len(alignment_problem_records)}")
    log("OK", f"RTC present cities: {rtc_present_city_count}")
    log("OK", f"RTC missing cities: {rtc_missing_city_count}")

    log("INFO", "Patch counts by region:")
    for region, count in sorted(patches_by_region.items()):
        log("INFO", f"  {region}: {count}")

    log("INFO", "Top patch counts by city:")
    for city, count in patches_by_city.most_common(10):
        log("INFO", f"  {city}: {count}")


if __name__ == "__main__":
    main()