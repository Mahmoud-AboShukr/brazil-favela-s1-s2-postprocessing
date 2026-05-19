#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_inventory_s1_rtc_raw.py

Inventory raw / processed Sentinel-1 RTC candidate files before integrating them into Instance C.

This script DOES NOT modify rasters.

It scans an RTC root folder, usually:

    D:/my_processed_data/s1_images

and compares it against the cities available in Instance C:

    D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired/s2_filled/<city>/

It produces inventory outputs under:

    <instance-root>/metadata/rtc_processing/

Main outputs:

    raw_rtc_inventory_files.csv
    raw_rtc_inventory_by_city.csv
    raw_rtc_inventory.json
    raw_rtc_inventory.md

Purpose:

Before we create:

    instance_C_s2_nodata_repaired/s1_rtc_ready/<city>/<city>_s1_rtc_vv_vh_10m_aligned.tif

we need to know:

    - how the RTC files are organized,
    - whether VV and VH are separate or stacked,
    - whether the files are already georeferenced,
    - their CRS, pixel size, shape, dtype, band count, and band descriptions,
    - which candidate files correspond to each city.

Example PowerShell command:

python src/rtc_processing/01_inventory_s1_rtc_raw.py `
  --rtc-root "D:/my_processed_data/s1_images" `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import rasterio
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


def normalize_text(value: str) -> str:
    value = value.lower()
    value = value.replace("\\", "/")
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = re.sub(r"[^a-z0-9_\/.]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value


def normalize_city_name(value: str) -> str:
    value = value.strip()
    value = value.replace("\\", "/").split("/")[-1]
    value = value.lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


# ---------------------------------------------------------------------
# City discovery
# ---------------------------------------------------------------------

def discover_instance_cities(instance_root: Path) -> List[str]:
    s2_root = instance_root / "s2_filled"

    if not s2_root.exists():
        fail(f"Could not find Instance C s2_filled folder: {path_to_str(s2_root)}")

    cities = [
        normalize_city_name(child.name)
        for child in sorted(s2_root.iterdir())
        if child.is_dir()
    ]

    if not cities:
        fail(f"No city folders found under: {path_to_str(s2_root)}")

    return cities


# ---------------------------------------------------------------------
# File scanning and classification
# ---------------------------------------------------------------------

def iter_raster_files(root: Path) -> Iterable[Path]:
    suffixes = {".tif", ".tiff"}

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def path_matches_city(path: Path, city: str) -> bool:
    text = normalize_text(path_to_str(path))
    city_norm = normalize_city_name(city)

    # Match city as a substring in the normalized full path.
    # This is intentionally permissive for first inventory.
    return city_norm in text


def find_matched_cities(path: Path, cities: Sequence[str]) -> List[str]:
    return [
        city for city in cities
        if path_matches_city(path, city)
    ]


def guess_role_from_name(path: Path, band_count: Optional[int]) -> str:
    """
    Guess whether a file is VV, VH, stacked VV/VH, or derived/unknown.

    This is only a heuristic. The output will be reviewed before final RTC processing.
    """

    stem = normalize_text(path.stem)
    full = normalize_text(path_to_str(path))

    has_vv = bool(re.search(r"(^|[_./])vv($|[_./])", full)) or "vv" in stem
    has_vh = bool(re.search(r"(^|[_./])vh($|[_./])", full)) or "vh" in stem

    derived_tokens = [
        "vvdiff",
        "vv_diff",
        "vvminusvh",
        "vv_minus_vh",
        "vv_vh_diff",
        "difference",
        "ratio",
    ]

    if any(token in full for token in derived_tokens):
        return "derived_ignore"

    if band_count is not None and band_count >= 2 and has_vv and has_vh:
        return "stacked_vv_vh"

    if band_count is not None and band_count >= 2 and ("rtc" in full or "terrain" in full):
        return "stacked_candidate"

    if has_vv and not has_vh:
        return "vv"

    if has_vh and not has_vv:
        return "vh"

    if has_vv and has_vh:
        return "vv_vh_name_ambiguous"

    return "unknown"


def score_candidate(path: str, role: str, city: str) -> Tuple[int, int, int, str]:
    """
    Lower tuple is better.

    Prefer:
        - role-specific files
        - files containing RTC / terrain / gamma / sigma clues
        - shorter paths
        - deterministic path tie-breaker
    """

    text = normalize_text(path)
    city_norm = normalize_city_name(city)

    role_priority = {
        "stacked_vv_vh": 0,
        "stacked_candidate": 1,
        "vv": 0,
        "vh": 0,
        "vv_vh_name_ambiguous": 2,
        "unknown": 5,
        "derived_ignore": 9,
    }.get(role, 5)

    rtc_bonus = 0
    if any(token in text for token in ["rtc", "terrain", "gamma0", "sigma0", "orthorectified", "corrected"]):
        rtc_bonus = -1

    city_bonus = 0 if city_norm in text else 1

    path_length = len(text)

    return (role_priority + rtc_bonus, city_bonus, path_length, text)


# ---------------------------------------------------------------------
# Raster metadata
# ---------------------------------------------------------------------

def raster_metadata(path: Path) -> Dict[str, object]:
    out: Dict[str, object] = {
        "open_status": "not_started",
        "open_error": "",
        "band_count": "",
        "width": "",
        "height": "",
        "dtype": "",
        "dtypes": "",
        "crs": "",
        "transform": "",
        "pixel_width": "",
        "pixel_height": "",
        "bounds_left": "",
        "bounds_bottom": "",
        "bounds_right": "",
        "bounds_top": "",
        "nodata": "",
        "band_descriptions": "",
        "driver": "",
        "compress": "",
    }

    try:
        with rasterio.open(path) as src:
            out["open_status"] = "ok"
            out["band_count"] = src.count
            out["width"] = src.width
            out["height"] = src.height
            out["dtype"] = src.dtypes[0] if src.dtypes else ""
            out["dtypes"] = ";".join(str(x) for x in src.dtypes)
            out["crs"] = str(src.crs) if src.crs is not None else ""
            out["transform"] = tuple(float(x) for x in src.transform)
            out["pixel_width"] = float(src.transform.a)
            out["pixel_height"] = float(src.transform.e)
            out["bounds_left"] = float(src.bounds.left)
            out["bounds_bottom"] = float(src.bounds.bottom)
            out["bounds_right"] = float(src.bounds.right)
            out["bounds_top"] = float(src.bounds.top)
            out["nodata"] = "" if src.nodata is None else src.nodata
            out["band_descriptions"] = ";".join(
                "" if desc is None else str(desc)
                for desc in src.descriptions
            )
            out["driver"] = src.driver
            out["compress"] = str(src.profile.get("compress", ""))

    except Exception as exc:
        out["open_status"] = "failed"
        out["open_error"] = repr(exc)

    return out


# ---------------------------------------------------------------------
# Inventory construction
# ---------------------------------------------------------------------

def build_file_inventory(
    rtc_root: Path,
    cities: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    raster_files = sorted(iter_raster_files(rtc_root))

    log("INFO", f"Found raster files under RTC root: {len(raster_files)}")

    for idx, path in enumerate(raster_files, start=1):
        if idx % 100 == 0:
            log("INFO", f"Inspected {idx}/{len(raster_files)} raster files")

        meta = raster_metadata(path)

        matched_cities = find_matched_cities(path, cities)
        matched_city_text = ";".join(matched_cities)

        band_count = None
        if meta["band_count"] != "":
            try:
                band_count = int(meta["band_count"])
            except Exception:
                band_count = None

        role_guess = guess_role_from_name(path, band_count)

        rel_path = ""
        try:
            rel_path = path_to_str(path.relative_to(rtc_root))
        except Exception:
            rel_path = path_to_str(path)

        row: Dict[str, object] = {
            "file_path": path_to_str(path),
            "relative_path": rel_path,
            "file_name": path.name,
            "file_stem": path.stem,
            "file_size_mb": round(path.stat().st_size / (1024 * 1024), 3),
            "matched_cities": matched_city_text,
            "matched_city_count": len(matched_cities),
            "role_guess": role_guess,
        }

        row.update(meta)

        rows.append(row)

    return rows


def choose_best_file(
    rows: List[Dict[str, object]],
    *,
    city: str,
    accepted_roles: Sequence[str],
) -> Optional[Dict[str, object]]:
    candidates = [
        row for row in rows
        if str(row.get("role_guess", "")) in set(accepted_roles)
    ]

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda row: score_candidate(
            str(row["file_path"]),
            str(row["role_guess"]),
            city,
        ),
    )

    return candidates[0]


def build_city_inventory(
    file_rows: List[Dict[str, object]],
    cities: Sequence[str],
) -> List[Dict[str, object]]:
    city_rows: List[Dict[str, object]] = []

    for city in cities:
        matched = [
            row for row in file_rows
            if city in str(row.get("matched_cities", "")).split(";")
        ]

        open_ok = [
            row for row in matched
            if row.get("open_status") == "ok"
        ]

        stacked_candidates = [
            row for row in open_ok
            if row.get("role_guess") in {"stacked_vv_vh", "stacked_candidate", "vv_vh_name_ambiguous"}
            and safe_float(row.get("band_count", 0)) >= 2
        ]

        vv_candidates = [
            row for row in open_ok
            if row.get("role_guess") == "vv"
        ]

        vh_candidates = [
            row for row in open_ok
            if row.get("role_guess") == "vh"
        ]

        derived_ignore = [
            row for row in open_ok
            if row.get("role_guess") == "derived_ignore"
        ]

        unknown = [
            row for row in open_ok
            if row.get("role_guess") == "unknown"
        ]

        best_stacked = choose_best_file(
            stacked_candidates,
            city=city,
            accepted_roles=["stacked_vv_vh", "stacked_candidate", "vv_vh_name_ambiguous"],
        )

        best_vv = choose_best_file(
            vv_candidates,
            city=city,
            accepted_roles=["vv"],
        )

        best_vh = choose_best_file(
            vh_candidates,
            city=city,
            accepted_roles=["vh"],
        )

        if best_stacked is not None:
            recommended_mode = "stacked"
        elif best_vv is not None and best_vh is not None:
            recommended_mode = "separate_vv_vh"
        else:
            recommended_mode = "missing_or_unclear"

        notes: List[str] = []

        if not matched:
            notes.append("No RTC candidate files matched this city name.")

        if matched and not open_ok:
            notes.append("Files matched city name, but none could be opened by rasterio.")

        if len(stacked_candidates) > 1:
            notes.append("Multiple stacked candidates found; selected one by heuristic.")

        if len(vv_candidates) > 1:
            notes.append("Multiple VV candidates found; selected one by heuristic.")

        if len(vh_candidates) > 1:
            notes.append("Multiple VH candidates found; selected one by heuristic.")

        if best_stacked is None and (best_vv is None or best_vh is None):
            notes.append("Could not identify a complete VV/VH RTC input for this city.")

        if derived_ignore:
            notes.append("Derived files such as VV_minus_VH/diff were detected and ignored.")

        city_row: Dict[str, object] = {
            "city": city,
            "n_matched_files": len(matched),
            "n_open_ok_files": len(open_ok),
            "n_stacked_candidates": len(stacked_candidates),
            "n_vv_candidates": len(vv_candidates),
            "n_vh_candidates": len(vh_candidates),
            "n_derived_ignored": len(derived_ignore),
            "n_unknown_candidates": len(unknown),
            "recommended_mode": recommended_mode,
            "recommended_stacked_path": path_to_str(Path(str(best_stacked["file_path"]))) if best_stacked else "",
            "recommended_vv_path": path_to_str(Path(str(best_vv["file_path"]))) if best_vv else "",
            "recommended_vh_path": path_to_str(Path(str(best_vh["file_path"]))) if best_vh else "",
            "notes": " | ".join(notes),
        }

        # Add basic geospatial metadata from the recommended file.
        ref = best_stacked or best_vv or best_vh

        if ref is not None:
            city_row.update(
                {
                    "recommended_band_count": ref.get("band_count", ""),
                    "recommended_width": ref.get("width", ""),
                    "recommended_height": ref.get("height", ""),
                    "recommended_crs": ref.get("crs", ""),
                    "recommended_pixel_width": ref.get("pixel_width", ""),
                    "recommended_pixel_height": ref.get("pixel_height", ""),
                    "recommended_dtype": ref.get("dtype", ""),
                    "recommended_nodata": ref.get("nodata", ""),
                    "recommended_band_descriptions": ref.get("band_descriptions", ""),
                }
            )
        else:
            city_row.update(
                {
                    "recommended_band_count": "",
                    "recommended_width": "",
                    "recommended_height": "",
                    "recommended_crs": "",
                    "recommended_pixel_width": "",
                    "recommended_pixel_height": "",
                    "recommended_dtype": "",
                    "recommended_nodata": "",
                    "recommended_band_descriptions": "",
                }
            )

        city_rows.append(city_row)

    return city_rows


# ---------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------

def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    if not rows:
        fail(f"No rows to write for CSV: {path_to_str(path)}")

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    payload: Dict[str, object],
    overwrite: bool,
) -> None:
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

    lines.append("# RTC raw inventory")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- RTC root: `{summary['rtc_root']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Cities expected: `{summary['n_cities_expected']}`")
    lines.append(f"- Raster files found: `{summary['n_raster_files_found']}`")
    lines.append(f"- Cities with complete recommended RTC input: `{summary['n_cities_complete_recommended']}`")
    lines.append(f"- Cities missing or unclear: `{summary['n_cities_missing_or_unclear']}`")
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    outputs = summary["outputs"]
    lines.append(f"- File inventory CSV: `{outputs['file_inventory_csv']}`")
    lines.append(f"- City inventory CSV: `{outputs['city_inventory_csv']}`")
    lines.append(f"- JSON: `{outputs['json']}`")
    lines.append(f"- Markdown: `{outputs['markdown']}`")
    lines.append("")

    lines.append("## City-level recommended RTC inputs")
    lines.append("")
    lines.append(
        "| city | mode | matched files | stacked | VV | VH | CRS | size | pixel size | notes |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---|---|---|---|"
    )

    for row in city_rows:
        size_text = ""
        if row["recommended_width"] != "" and row["recommended_height"] != "":
            size_text = f"{row['recommended_width']}×{row['recommended_height']}"

        pixel_text = ""
        if row["recommended_pixel_width"] != "" and row["recommended_pixel_height"] != "":
            pixel_text = f"{row['recommended_pixel_width']}, {row['recommended_pixel_height']}"

        lines.append(
            f"| {row['city']} | "
            f"{row['recommended_mode']} | "
            f"{row['n_matched_files']} | "
            f"{row['n_stacked_candidates']} | "
            f"{row['n_vv_candidates']} | "
            f"{row['n_vh_candidates']} | "
            f"{row['recommended_crs']} | "
            f"{size_text} | "
            f"{pixel_text} | "
            f"{row['notes']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `stacked` means the script found a multi-band candidate that likely contains VV/VH.")
    lines.append("- `separate_vv_vh` means the script found separate VV and VH files.")
    lines.append("- `missing_or_unclear` means the city needs manual inspection before RTC alignment.")
    lines.append("- Derived files such as VV_minus_VH are intentionally ignored for the CROMA comparison.")
    lines.append("- This script does not align, resample, repair, or write RTC rasters into Instance C.")
    lines.append("- The next step is to create `s1_rtc_ready/` using S2 as the reference grid.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory raw/processed S1 RTC candidate files before Instance C integration."
    )

    parser.add_argument(
        "--rtc-root",
        type=Path,
        required=True,
        help="Root folder containing raw/processed RTC Sentinel-1 files.",
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing inventory outputs.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    rtc_root: Path = args.rtc_root
    instance_root: Path = args.instance_root

    output_dir: Path = args.output_dir or (
        instance_root / "metadata" / "rtc_processing"
    )

    file_inventory_csv = output_dir / "raw_rtc_inventory_files.csv"
    city_inventory_csv = output_dir / "raw_rtc_inventory_by_city.csv"
    json_path = output_dir / "raw_rtc_inventory.json"
    md_path = output_dir / "raw_rtc_inventory.md"

    log("STEP", "Starting RTC raw inventory.")
    log("INFO", f"RTC root:      {path_to_str(rtc_root)}")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")

    if not rtc_root.exists():
        fail(f"RTC root does not exist: {path_to_str(rtc_root)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    cities = discover_instance_cities(instance_root)

    log("OK", f"Discovered Instance C cities: {len(cities)}")

    file_rows = build_file_inventory(rtc_root, cities)
    city_rows = build_city_inventory(file_rows, cities)

    n_complete = sum(
        1 for row in city_rows
        if row["recommended_mode"] in {"stacked", "separate_vv_vh"}
    )

    n_missing_or_unclear = sum(
        1 for row in city_rows
        if row["recommended_mode"] == "missing_or_unclear"
    )

    summary: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rtc_root": path_to_str(rtc_root),
        "instance_root": path_to_str(instance_root),
        "n_cities_expected": len(cities),
        "cities": cities,
        "n_raster_files_found": len(file_rows),
        "n_cities_complete_recommended": n_complete,
        "n_cities_missing_or_unclear": n_missing_or_unclear,
        "outputs": {
            "file_inventory_csv": path_to_str(file_inventory_csv),
            "city_inventory_csv": path_to_str(city_inventory_csv),
            "json": path_to_str(json_path),
            "markdown": path_to_str(md_path),
        },
        "city_inventory": city_rows,
    }

    log("STEP", "Writing inventory outputs.")

    write_csv(file_inventory_csv, file_rows, overwrite=args.overwrite)
    write_csv(city_inventory_csv, city_rows, overwrite=args.overwrite)
    write_json(json_path, summary, overwrite=args.overwrite)
    write_markdown(md_path, summary, city_rows, overwrite=args.overwrite)

    log("OK", f"Wrote file inventory CSV: {path_to_str(file_inventory_csv)}")
    log("OK", f"Wrote city inventory CSV: {path_to_str(city_inventory_csv)}")
    log("OK", f"Wrote JSON: {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown: {path_to_str(md_path)}")

    log("STEP", "Final summary.")
    log("OK", f"Raster files found: {len(file_rows)}")
    log("OK", f"Cities expected: {len(cities)}")
    log("OK", f"Cities with complete recommended RTC input: {n_complete}")
    log("OK", f"Cities missing or unclear: {n_missing_or_unclear}")

    if n_missing_or_unclear > 0:
        log("WARN", "Some cities are missing or unclear. Inspect the Markdown/CSV before finalizing RTC.")


if __name__ == "__main__":
    main()