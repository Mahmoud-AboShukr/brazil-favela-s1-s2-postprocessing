#!/usr/bin/env python3
"""
Build an inventory of available Sentinel-1 SAFE products for SNAP preprocessing.

Purpose
-------
This script checks whether we have full Sentinel-1 SAFE products available for
each city. SNAP terrain correction requires a full Sentinel-1 product, usually:

    S1A_... .SAFE/
    S1B_... .SAFE/
    S1C_... .SAFE/

or sometimes the zipped equivalent:

    S1A_... .SAFE.zip

The current direct-GCP baseline only needs VV.tif and VH.tif, but the SNAP
pipeline needs the full SAFE product.

Input assumptions
-----------------
The config contains:

    output_root: D:/post_processing_dataset
    s1_root: D:/my_processed_data/s1_images/raw

The finalized city list is discovered from:

    <output_root>/s2_final/<city>/

The current raw S1 folders are checked under:

    <s1_root>/<city>/**/VV.tif
    <s1_root>/<city>/**/VH.tif
    <s1_root>/<city>/**/selected_s1_item.json

SAFE products are searched under likely locations, including:

    <input_root>/s1_images/safe
    <input_root>/s1_images/raw_safe
    <input_root>/s1_images/safe_raw
    <input_root>/s1_images/safe_products
    <input_root>/sentinel1_safe
    <input_root>/s1_safe
    <s1_root>
    <s1_root parent>

You can also pass explicit SAFE roots using:

    --safe-root D:/some/path/with/safe/products

Outputs
-------
Main inventory:

    <output_root>/metadata/s1_safe_inventory.csv

All detected SAFE candidates:

    <output_root>/metadata/s1_safe_candidates.csv

Summary:

    <output_root>/qc/s1_safe_inventory_summary.csv

Example
-------
Run all cities:

    python src/favela_postprocessing/07a_build_s1_safe_inventory.py --config configs/default.yaml

Run only pilot cities:

    python src/favela_postprocessing/07a_build_s1_safe_inventory.py --config configs/default.yaml --pilot-only

Run with an explicit SAFE folder:

    python src/favela_postprocessing/07a_build_s1_safe_inventory.py --config configs/default.yaml --safe-root D:/my_processed_data/s1_images/safe
"""

from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from tqdm import tqdm


SCRIPT_NAME = "07a_build_s1_safe_inventory.py"

PILOT_CITIES = [
    "rio_de_janeiro",
    "belem",
    "porto_alegre",
]

S1_PRODUCT_RE = re.compile(
    r"^(?P<platform>S1[A-Z])_"
    r"(?P<mode>[A-Z0-9]+)_"
    r"(?P<product_type>[A-Z0-9]+)_"
    r"(?P<resolution_class>[A-Z0-9])"
    r"(?P<processing_level>[A-Z0-9])"
    r"(?P<product_class>[A-Z0-9])"
    r"(?P<polarization>[A-Z0-9]+)_"
    r"(?P<start>\d{8}T\d{6})_"
    r"(?P<stop>\d{8}T\d{6})_"
    r"(?P<absolute_orbit>\d+)_"
    r"(?P<datatake>[A-Fa-f0-9]+)"
    r"(?:_(?P<unique_id>[A-Fa-f0-9]+))?$"
)


@dataclass
class SafeCandidate:
    product_id: str
    path: Path
    kind: str
    has_manifest: bool
    measurement_file_count: int
    has_measurement_vv: bool
    has_measurement_vh: bool
    is_valid_for_snap: bool
    platform: Optional[str]
    acquisition_mode: Optional[str]
    product_type: Optional[str]
    product_class: Optional[str]
    polarization_code: Optional[str]
    acquisition_start: Optional[str]
    acquisition_stop: Optional[str]
    absolute_orbit: Optional[str]
    datatake: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Sentinel-1 SAFE inventory for SNAP preprocessing."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--safe-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Root folder to search for Sentinel-1 SAFE products. "
            "Can be repeated. If omitted, likely roots are inferred."
        ),
    )
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="Process only one city. Can be repeated.",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Only process the pilot cities: rio_de_janeiro, belem, porto_alegre.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Maximum folder depth to search below each SAFE root. Default: 8",
    )
    parser.add_argument(
        "--skip-zips",
        action="store_true",
        help="Do not inspect .zip SAFE products.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    required_keys = ["output_root", "s1_root"]
    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"Missing required key in config: {key}")

    return cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_city_name(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def discover_city_list(
    output_root: Path,
    selected_cities: Optional[Sequence[str]],
    pilot_only: bool,
) -> List[str]:
    if pilot_only:
        return sorted(PILOT_CITIES)

    if selected_cities:
        return sorted(set(normalize_city_name(city) for city in selected_cities))

    s2_final_root = output_root / "s2_final"

    if not s2_final_root.exists():
        raise FileNotFoundError(
            f"Could not discover cities because s2_final root does not exist: {s2_final_root}"
        )

    cities = sorted(path.name for path in s2_final_root.iterdir() if path.is_dir())

    if not cities:
        raise RuntimeError(f"No city folders found under: {s2_final_root}")

    return cities


def candidate_safe_roots(cfg: Dict[str, Any], explicit_roots: Optional[Sequence[Path]]) -> List[Path]:
    roots: List[Path] = []

    if explicit_roots:
        roots.extend(explicit_roots)

    input_root = Path(str(cfg.get("input_root", ""))) if cfg.get("input_root") else None
    s1_root = Path(str(cfg["s1_root"]))

    if cfg.get("s1_safe_root"):
        roots.append(Path(str(cfg["s1_safe_root"])))

    if input_root:
        roots.extend(
            [
                input_root / "s1_images" / "safe",
                input_root / "s1_images" / "raw_safe",
                input_root / "s1_images" / "safe_raw",
                input_root / "s1_images" / "safe_products",
                input_root / "sentinel1_safe",
                input_root / "s1_safe",
                input_root / "S1_SAFE",
            ]
        )

    roots.extend(
        [
            s1_root,
            s1_root.parent,
        ]
    )

    unique_existing_roots: List[Path] = []
    seen: set[str] = set()

    for root in roots:
        root = Path(root)
        try:
            resolved = str(root.resolve())
        except Exception:
            resolved = str(root)

        if resolved in seen:
            continue

        seen.add(resolved)

        if root.exists() and root.is_dir():
            unique_existing_roots.append(root)

    return unique_existing_roots


def extract_product_id_from_path(path: Path) -> Optional[str]:
    name = path.name

    lower_name = name.lower()

    if lower_name.endswith(".safe"):
        return name[:-5]

    if lower_name.endswith(".safe.zip"):
        return name[:-9]

    if lower_name.endswith(".zip"):
        stem = name[:-4]
        if stem.lower().endswith(".safe"):
            stem = stem[:-5]
        return stem

    if name.startswith("S1"):
        return name

    return None


def parse_s1_product_id(product_id: str) -> Dict[str, Optional[str]]:
    match = S1_PRODUCT_RE.match(product_id)

    if not match:
        return {
            "platform": None,
            "acquisition_mode": None,
            "product_type": None,
            "product_class": None,
            "polarization_code": None,
            "acquisition_start": None,
            "acquisition_stop": None,
            "absolute_orbit": None,
            "datatake": None,
        }

    groups = match.groupdict()

    start_raw = groups.get("start")
    stop_raw = groups.get("stop")

    def parse_datetime(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%S").isoformat()
        except ValueError:
            return value

    return {
        "platform": groups.get("platform"),
        "acquisition_mode": groups.get("mode"),
        "product_type": groups.get("product_type"),
        "product_class": groups.get("product_class"),
        "polarization_code": groups.get("polarization"),
        "acquisition_start": parse_datetime(start_raw),
        "acquisition_stop": parse_datetime(stop_raw),
        "absolute_orbit": groups.get("absolute_orbit"),
        "datatake": groups.get("datatake"),
    }


def measurement_flags_from_names(names: Iterable[str]) -> Tuple[int, bool, bool]:
    measurement_names = []

    for name in names:
        lower = name.replace("\\", "/").lower()
        if "/measurement/" in lower and (lower.endswith(".tif") or lower.endswith(".tiff")):
            measurement_names.append(lower)

    has_vv = any("vv" in Path(name).name.lower() for name in measurement_names)
    has_vh = any("vh" in Path(name).name.lower() for name in measurement_names)

    return len(measurement_names), has_vv, has_vh


def inspect_safe_directory(path: Path) -> Tuple[bool, int, bool, bool]:
    manifest_path = path / "manifest.safe"
    has_manifest = manifest_path.exists()

    measurement_dir = path / "measurement"
    measurement_files: List[Path] = []

    if measurement_dir.exists() and measurement_dir.is_dir():
        for item in measurement_dir.iterdir():
            if item.is_file() and item.suffix.lower() in [".tif", ".tiff"]:
                measurement_files.append(item)

    has_vv = any("vv" in item.name.lower() for item in measurement_files)
    has_vh = any("vh" in item.name.lower() for item in measurement_files)

    return has_manifest, len(measurement_files), has_vv, has_vh


def inspect_safe_zip(path: Path) -> Tuple[bool, int, bool, bool]:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
    except Exception:
        return False, 0, False, False

    has_manifest = any(name.replace("\\", "/").lower().endswith("/manifest.safe") for name in names)

    if not has_manifest:
        has_manifest = any(name.replace("\\", "/").lower() == "manifest.safe" for name in names)

    measurement_file_count, has_vv, has_vh = measurement_flags_from_names(names)

    return has_manifest, measurement_file_count, has_vv, has_vh


def make_safe_candidate(path: Path) -> Optional[SafeCandidate]:
    product_id = extract_product_id_from_path(path)

    if not product_id:
        return None

    lower_name = path.name.lower()

    if lower_name.endswith(".safe") and path.is_dir():
        kind = "directory_safe"
        has_manifest, measurement_count, has_vv, has_vh = inspect_safe_directory(path)
    elif lower_name.endswith(".zip") and path.is_file():
        kind = "zip_safe"
        has_manifest, measurement_count, has_vv, has_vh = inspect_safe_zip(path)
    else:
        return None

    parsed = parse_s1_product_id(product_id)

    is_valid_for_snap = bool(has_manifest and measurement_count > 0)

    return SafeCandidate(
        product_id=product_id,
        path=path,
        kind=kind,
        has_manifest=has_manifest,
        measurement_file_count=measurement_count,
        has_measurement_vv=has_vv,
        has_measurement_vh=has_vh,
        is_valid_for_snap=is_valid_for_snap,
        platform=parsed["platform"],
        acquisition_mode=parsed["acquisition_mode"],
        product_type=parsed["product_type"],
        product_class=parsed["product_class"],
        polarization_code=parsed["polarization_code"],
        acquisition_start=parsed["acquisition_start"],
        acquisition_stop=parsed["acquisition_stop"],
        absolute_orbit=parsed["absolute_orbit"],
        datatake=parsed["datatake"],
    )


def discover_safe_candidates(
    roots: Sequence[Path],
    max_depth: int,
    include_zips: bool,
) -> List[SafeCandidate]:
    candidates: List[SafeCandidate] = []
    seen_paths: set[str] = set()

    for root in roots:
        print(f"[INFO] Scanning SAFE root: {root}")

        root = root.resolve()
        root_depth = len(root.parts)

        for current_dir, dirnames, filenames in os.walk(root):
            current_path = Path(current_dir)
            current_depth = len(current_path.resolve().parts) - root_depth

            safe_dirnames = [
                dirname
                for dirname in list(dirnames)
                if dirname.lower().endswith(".safe") and dirname.upper().startswith("S1")
            ]

            for dirname in safe_dirnames:
                safe_path = current_path / dirname
                safe_key = str(safe_path.resolve())

                if safe_key not in seen_paths:
                    candidate = make_safe_candidate(safe_path)
                    if candidate:
                        candidates.append(candidate)
                        seen_paths.add(safe_key)

                dirnames.remove(dirname)

            if include_zips:
                for filename in filenames:
                    lower = filename.lower()
                    upper = filename.upper()

                    if not upper.startswith("S1"):
                        continue

                    if not lower.endswith(".zip"):
                        continue

                    zip_path = current_path / filename
                    zip_key = str(zip_path.resolve())

                    if zip_key in seen_paths:
                        continue

                    candidate = make_safe_candidate(zip_path)
                    if candidate:
                        candidates.append(candidate)
                        seen_paths.add(zip_key)

            if current_depth >= max_depth:
                dirnames[:] = []

    candidates = sorted(candidates, key=lambda item: (item.product_id, str(item.path)))

    return candidates


def read_selected_s1_item_ids(city_root: Path) -> List[str]:
    ids: List[str] = []

    if not city_root.exists():
        return ids

    for json_path in city_root.rglob("selected_s1_item.json"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        item_id = data.get("id")

        if isinstance(item_id, str) and item_id:
            ids.append(item_id)
            continue

        properties = data.get("properties", {})
        if isinstance(properties, dict):
            for key in ["s1:product_identifier", "safe:product_id", "product_id"]:
                value = properties.get(key)
                if isinstance(value, str) and value:
                    ids.append(value)

    return sorted(set(ids))


def find_raw_s1_files(city_root: Path) -> Dict[str, Any]:
    raw_vv_paths: List[Path] = []
    raw_vh_paths: List[Path] = []
    selected_json_paths: List[Path] = []

    if city_root.exists():
        for path in city_root.rglob("*"):
            if not path.is_file():
                continue

            lower_name = path.name.lower()

            if lower_name == "vv.tif":
                raw_vv_paths.append(path)
            elif lower_name == "vh.tif":
                raw_vh_paths.append(path)
            elif lower_name == "selected_s1_item.json":
                selected_json_paths.append(path)

    product_dirs = sorted(
        set(str(path.parent) for path in raw_vv_paths + raw_vh_paths + selected_json_paths)
    )

    return {
        "has_raw_vv": len(raw_vv_paths) > 0,
        "has_raw_vh": len(raw_vh_paths) > 0,
        "raw_vv_count": len(raw_vv_paths),
        "raw_vh_count": len(raw_vh_paths),
        "selected_json_count": len(selected_json_paths),
        "raw_vv_path": str(raw_vv_paths[0]) if raw_vv_paths else "",
        "raw_vh_path": str(raw_vh_paths[0]) if raw_vh_paths else "",
        "selected_json_path": str(selected_json_paths[0]) if selected_json_paths else "",
        "raw_product_dirs": " | ".join(product_dirs),
    }


def path_parts_normalized(path: Path) -> List[str]:
    return [normalize_city_name(part) for part in path.parts]


def score_candidate_for_city(
    candidate: SafeCandidate,
    city: str,
    selected_item_ids: Sequence[str],
) -> Tuple[int, str]:
    score = 0
    reasons: List[str] = []

    normalized_city = normalize_city_name(city)
    normalized_parts = path_parts_normalized(candidate.path)

    if normalized_city in normalized_parts:
        score += 50
        reasons.append("city_name_in_path")

    if candidate.product_id in selected_item_ids:
        score += 100
        reasons.append("matches_selected_s1_item_id")

    for item_id in selected_item_ids:
        if candidate.product_id in item_id or item_id in candidate.product_id:
            score += 80
            reasons.append("partially_matches_selected_s1_item_id")
            break

    if candidate.has_measurement_vv and candidate.has_measurement_vh:
        score += 20
        reasons.append("has_vv_vh_measurements")
    elif candidate.polarization_code and "DV" in candidate.polarization_code:
        score += 15
        reasons.append("dual_pol_dv_in_product_id")

    if candidate.is_valid_for_snap:
        score += 20
        reasons.append("valid_for_snap")

    if candidate.kind == "directory_safe":
        score += 5
        reasons.append("unpacked_safe_directory")

    if not reasons:
        reasons.append("no_direct_match")

    return score, "+".join(reasons)


def select_best_candidate_for_city(
    city: str,
    candidates: Sequence[SafeCandidate],
    selected_item_ids: Sequence[str],
) -> Tuple[Optional[SafeCandidate], str, List[Tuple[SafeCandidate, int, str]]]:
    scored: List[Tuple[SafeCandidate, int, str]] = []

    for candidate in candidates:
        score, reason = score_candidate_for_city(candidate, city, selected_item_ids)
        if score > 0:
            scored.append((candidate, score, reason))

    scored = sorted(
        scored,
        key=lambda item: (
            item[1],
            item[0].is_valid_for_snap,
            item[0].has_measurement_vv and item[0].has_measurement_vh,
            item[0].kind == "directory_safe",
        ),
        reverse=True,
    )

    if not scored:
        return None, "no_safe_candidate_match", []

    best_candidate, best_score, best_reason = scored[0]

    return best_candidate, best_reason, scored


def determine_status(
    has_raw_vv: bool,
    has_raw_vh: bool,
    best_candidate: Optional[SafeCandidate],
) -> Tuple[str, str]:
    has_raw_pair = has_raw_vv and has_raw_vh

    if best_candidate and best_candidate.is_valid_for_snap:
        if has_raw_pair:
            return (
                "SAFE_READY",
                "Full SAFE product found and raw VV/VH pair exists. Ready for SNAP pilot.",
            )
        return (
            "SAFE_FOUND_RAW_INCOMPLETE",
            "Full SAFE product found, but current raw VV/VH pair is incomplete or missing.",
        )

    if best_candidate and not best_candidate.is_valid_for_snap:
        return (
            "SAFE_FOUND_BUT_INCOMPLETE",
            "A SAFE-like product was found, but manifest/measurement files are incomplete. Inspect or re-extract it.",
        )

    if has_raw_pair:
        return (
            "RAW_VV_VH_ONLY_NO_SAFE",
            "Only VV/VH GeoTIFFs found. Good for direct-GCP baseline, but not enough for SNAP terrain correction.",
        )

    return (
        "MISSING_SAFE_AND_RAW_INCOMPLETE",
        "No valid SAFE product found and raw VV/VH pair is incomplete.",
    )


def candidate_to_row(candidate: SafeCandidate) -> Dict[str, Any]:
    return {
        "safe_product_id": candidate.product_id,
        "safe_path": str(candidate.path),
        "safe_kind": candidate.kind,
        "safe_has_manifest": candidate.has_manifest,
        "safe_measurement_file_count": candidate.measurement_file_count,
        "safe_has_measurement_vv": candidate.has_measurement_vv,
        "safe_has_measurement_vh": candidate.has_measurement_vh,
        "safe_is_valid_for_snap": candidate.is_valid_for_snap,
        "platform": candidate.platform,
        "acquisition_mode": candidate.acquisition_mode,
        "product_type": candidate.product_type,
        "product_class": candidate.product_class,
        "polarization_code": candidate.polarization_code,
        "acquisition_start": candidate.acquisition_start,
        "acquisition_stop": candidate.acquisition_stop,
        "absolute_orbit": candidate.absolute_orbit,
        "datatake": candidate.datatake,
    }


def build_inventory_rows(
    cities: Sequence[str],
    s1_root: Path,
    candidates: Sequence[SafeCandidate],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    inventory_rows: List[Dict[str, Any]] = []
    city_candidate_rows: List[Dict[str, Any]] = []

    for city in tqdm(cities, desc="Building S1 SAFE inventory"):
        city_root = s1_root / city

        raw_info = find_raw_s1_files(city_root)
        selected_item_ids = read_selected_s1_item_ids(city_root)

        best_candidate, match_reason, scored_candidates = select_best_candidate_for_city(
            city=city,
            candidates=candidates,
            selected_item_ids=selected_item_ids,
        )

        for candidate, score, reason in scored_candidates:
            row = {
                "city": city,
                "match_score": score,
                "match_reason": reason,
                "selected_s1_item_ids": " | ".join(selected_item_ids),
            }
            row.update(candidate_to_row(candidate))
            city_candidate_rows.append(row)

        status, recommendation = determine_status(
            has_raw_vv=raw_info["has_raw_vv"],
            has_raw_vh=raw_info["has_raw_vh"],
            best_candidate=best_candidate,
        )

        row: Dict[str, Any] = {
            "city": city,
            "status": status,
            "recommendation": recommendation,
            "city_s1_raw_root": str(city_root),
            "selected_s1_item_ids": " | ".join(selected_item_ids),
            "selected_s1_item_count": len(selected_item_ids),
            "matched_by": match_reason,
            "candidate_match_count": len(scored_candidates),
            "all_candidate_product_ids": " | ".join(
                candidate.product_id for candidate, _, _ in scored_candidates
            ),
        }

        row.update(raw_info)

        if best_candidate:
            row.update(candidate_to_row(best_candidate))
        else:
            row.update(
                {
                    "safe_product_id": "",
                    "safe_path": "",
                    "safe_kind": "",
                    "safe_has_manifest": False,
                    "safe_measurement_file_count": 0,
                    "safe_has_measurement_vv": False,
                    "safe_has_measurement_vh": False,
                    "safe_is_valid_for_snap": False,
                    "platform": "",
                    "acquisition_mode": "",
                    "product_type": "",
                    "product_class": "",
                    "polarization_code": "",
                    "acquisition_start": "",
                    "acquisition_stop": "",
                    "absolute_orbit": "",
                    "datatake": "",
                }
            )

        inventory_rows.append(row)

    return inventory_rows, city_candidate_rows


def write_outputs(
    output_root: Path,
    inventory_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
) -> None:
    metadata_root = output_root / "metadata"
    qc_root = output_root / "qc"

    ensure_dir(metadata_root)
    ensure_dir(qc_root)

    inventory_csv = metadata_root / "s1_safe_inventory.csv"
    candidates_csv = metadata_root / "s1_safe_candidates.csv"
    summary_csv = qc_root / "s1_safe_inventory_summary.csv"

    inventory_df = pd.DataFrame(inventory_rows)
    candidate_df = pd.DataFrame(candidate_rows)

    inventory_df.to_csv(inventory_csv, index=False)
    candidate_df.to_csv(candidates_csv, index=False)

    summary_df = (
        inventory_df["status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="city_count")
    )
    summary_df.to_csv(summary_csv, index=False)

    print(f"[INFO] Wrote inventory: {inventory_csv}")
    print(f"[INFO] Wrote SAFE candidates: {candidates_csv}")
    print(f"[INFO] Wrote summary: {summary_csv}")

    print("[INFO] Status counts:")
    print(summary_df.to_string(index=False))

    pilot_df = inventory_df[inventory_df["city"].isin(PILOT_CITIES)]

    if not pilot_df.empty:
        print("[INFO] Pilot city status:")
        print(
            pilot_df[
                [
                    "city",
                    "status",
                    "safe_product_id",
                    "safe_kind",
                    "safe_is_valid_for_snap",
                    "has_raw_vv",
                    "has_raw_vh",
                ]
            ].to_string(index=False)
        )


def main() -> int:
    args = parse_args()

    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))
    s1_root = Path(str(cfg["s1_root"]))

    include_zips = not args.skip_zips

    cities = discover_city_list(
        output_root=output_root,
        selected_cities=args.city,
        pilot_only=args.pilot_only,
    )

    roots = candidate_safe_roots(cfg=cfg, explicit_roots=args.safe_root)

    print("[INFO] Sentinel-1 SAFE inventory")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] S1 raw root: {s1_root}")
    print(f"[INFO] Cities to check: {len(cities)}")
    print(f"[INFO] Include ZIP SAFE products: {include_zips}")
    print(f"[INFO] Max SAFE search depth: {args.max_depth}")

    if not roots:
        print("[WARN] No existing SAFE search roots found.")
        print("[WARN] Inventory will still check raw VV/VH files, but SAFE status will be missing.")

    print("[INFO] SAFE search roots:")
    for root in roots:
        print(f"       - {root}")

    candidates = discover_safe_candidates(
        roots=roots,
        max_depth=args.max_depth,
        include_zips=include_zips,
    )

    print(f"[INFO] Total SAFE-like candidates found: {len(candidates)}")

    inventory_rows, candidate_rows = build_inventory_rows(
        cities=cities,
        s1_root=s1_root,
        candidates=candidates,
    )

    write_outputs(
        output_root=output_root,
        inventory_rows=inventory_rows,
        candidate_rows=candidate_rows,
    )

    inventory_df = pd.DataFrame(inventory_rows)

    safe_ready_count = int((inventory_df["status"] == "SAFE_READY").sum())
    raw_only_count = int((inventory_df["status"] == "RAW_VV_VH_ONLY_NO_SAFE").sum())

    print(f"[INFO] SAFE-ready cities: {safe_ready_count}")
    print(f"[INFO] Raw VV/VH only cities: {raw_only_count}")

    if safe_ready_count == 0:
        print("[WARN] No city is currently SAFE-ready for SNAP preprocessing.")
        print("[WARN] You likely need to download full Sentinel-1 SAFE products first.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())