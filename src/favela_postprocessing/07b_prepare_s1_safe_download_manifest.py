#!/usr/bin/env python3
"""
Prepare a Sentinel-1 SAFE download manifest from the already-selected raw S1 scenes.

Purpose
-------
The current project already has Sentinel-1 raw VV.tif and VH.tif files for each city:

    <s1_root>/<city>/<selected_product_id>/VV.tif
    <s1_root>/<city>/<selected_product_id>/VH.tif
    <s1_root>/<city>/<selected_product_id>/selected_s1_item.json

These files are enough for the direct-GCP baseline, but not enough for SNAP terrain correction.

SNAP needs the full Sentinel-1 SAFE product:

    S1A_....SAFE/
    S1B_....SAFE/
    S1A_....SAFE.zip
    S1B_....SAFE.zip

This script reads the existing selected_s1_item.json files and creates a clean manifest
listing exactly which full SAFE products should be downloaded.

Inputs
------
Config keys:
    input_root
    output_root
    s1_root

Main input:
    <s1_root>/<city>/**/selected_s1_item.json

Outputs
-------
Full manifest:
    <output_root>/metadata/s1_safe_download_manifest.csv

Pilot-only manifest:
    <output_root>/metadata/s1_safe_download_manifest_pilot.csv

Plain text list of SAFE product names:
    <output_root>/metadata/s1_safe_product_names_to_download.txt

Pilot-only plain text list:
    <output_root>/metadata/s1_safe_product_names_to_download_pilot.txt

Summary:
    <output_root>/qc/s1_safe_download_manifest_summary.csv

Example
-------
All cities:

    python src/favela_postprocessing/07b_prepare_s1_safe_download_manifest.py --config configs/default.yaml

Pilot cities only:

    python src/favela_postprocessing/07b_prepare_s1_safe_download_manifest.py --config configs/default.yaml --pilot-only

Custom SAFE target root:

    python src/favela_postprocessing/07b_prepare_s1_safe_download_manifest.py --config configs/default.yaml --safe-root D:/my_processed_data/s1_images/safe
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from tqdm import tqdm


SCRIPT_NAME = "07b_prepare_s1_safe_download_manifest.py"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Sentinel-1 SAFE download manifest from selected S1 items."
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
        default=None,
        help=(
            "Target root where downloaded SAFE products should be stored. "
            "Default: <input_root>/s1_images/safe"
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
        help="Only prepare manifest for rio_de_janeiro, belem, porto_alegre.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    required = ["input_root", "output_root", "s1_root"]

    for key in required:
        if key not in cfg:
            raise KeyError(f"Missing required key in config: {key}")

    return cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_city_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def discover_cities(
    output_root: Path,
    s1_root: Path,
    selected_cities: Optional[Sequence[str]],
    pilot_only: bool,
) -> List[str]:
    """
    Discover the official city list for SAFE download manifest preparation.

    Important:
    ----------
    We intentionally discover the full city list from:

        <output_root>/s2_final/

    rather than:

        <s1_root>/

    because the raw S1 folder may contain non-city folders such as:

        ocm_v1

    The finalized S2 folder is the clean 26-city reference list for the dataset.
    """
    if pilot_only:
        return sorted(PILOT_CITIES)

    if selected_cities:
        cities = sorted(set(normalize_city_name(city) for city in selected_cities))

        missing_raw = [
            city for city in cities
            if not (s1_root / city).exists()
        ]

        if missing_raw:
            print("[WARN] These selected cities do not have a matching S1 raw folder:")
            for city in missing_raw:
                print(f"       - {city}: expected {s1_root / city}")

        return cities

    s2_final_root = output_root / "s2_final"

    if not s2_final_root.exists():
        raise FileNotFoundError(
            f"Could not discover official city list because s2_final root does not exist: "
            f"{s2_final_root}"
        )

    cities = sorted(
        path.name
        for path in s2_final_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )

    if not cities:
        raise RuntimeError(f"No city folders found under finalized S2 root: {s2_final_root}")

    missing_raw = [
        city for city in cities
        if not (s1_root / city).exists()
    ]

    if missing_raw:
        print("[WARN] These official cities do not have a matching S1 raw folder:")
        for city in missing_raw:
            print(f"       - {city}: expected {s1_root / city}")

    return cities

def parse_product_datetime(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    value = str(value)

    for fmt in ["%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"]:
        try:
            return datetime.strptime(value, fmt).isoformat()
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return value


def clean_product_id(value: str) -> str:
    product_id = str(value).strip()

    if product_id.endswith(".SAFE.zip"):
        product_id = product_id[:-9]

    if product_id.endswith(".SAFE"):
        product_id = product_id[:-5]

    if product_id.endswith(".zip"):
        product_id = product_id[:-4]

    return product_id


def safe_name_from_product_id(product_id: str) -> str:
    return f"{clean_product_id(product_id)}.SAFE"


def safe_zip_name_from_product_id(product_id: str) -> str:
    return f"{clean_product_id(product_id)}.SAFE.zip"


def parse_s1_product_id(product_id: str) -> Dict[str, Optional[str]]:
    product_id = clean_product_id(product_id)

    match = S1_PRODUCT_RE.match(product_id)

    if not match:
        return {
            "platform_from_name": None,
            "acquisition_mode_from_name": None,
            "product_type_from_name": None,
            "resolution_class_from_name": None,
            "processing_level_from_name": None,
            "product_class_from_name": None,
            "polarization_code_from_name": None,
            "acquisition_start_from_name": None,
            "acquisition_stop_from_name": None,
            "absolute_orbit_from_name": None,
            "datatake_from_name": None,
        }

    groups = match.groupdict()

    return {
        "platform_from_name": groups.get("platform"),
        "acquisition_mode_from_name": groups.get("mode"),
        "product_type_from_name": groups.get("product_type"),
        "resolution_class_from_name": groups.get("resolution_class"),
        "processing_level_from_name": groups.get("processing_level"),
        "product_class_from_name": groups.get("product_class"),
        "polarization_code_from_name": groups.get("polarization"),
        "acquisition_start_from_name": parse_product_datetime(groups.get("start")),
        "acquisition_stop_from_name": parse_product_datetime(groups.get("stop")),
        "absolute_orbit_from_name": groups.get("absolute_orbit"),
        "datatake_from_name": groups.get("datatake"),
    }


def first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue

        value_str = str(value).strip()

        if value_str:
            return value_str

    return ""


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(data: Dict[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def extract_asset_hrefs(data: Dict[str, Any]) -> Dict[str, str]:
    assets = data.get("assets", {})

    if not isinstance(assets, dict):
        return {}

    hrefs: Dict[str, str] = {}

    for key, value in assets.items():
        if not isinstance(value, dict):
            continue

        href = value.get("href")

        if isinstance(href, str) and href:
            hrefs[str(key)] = href

    return hrefs


def find_raw_files(product_dir: Path) -> Dict[str, str]:
    vv_paths = sorted(product_dir.rglob("VV.tif"))
    vh_paths = sorted(product_dir.rglob("VH.tif"))

    return {
        "raw_vv_path": str(vv_paths[0]) if vv_paths else "",
        "raw_vh_path": str(vh_paths[0]) if vh_paths else "",
        "has_raw_vv": bool(vv_paths),
        "has_raw_vh": bool(vh_paths),
    }


def infer_city_product_dir(selected_json_path: Path, city_root: Path) -> Path:
    """
    Usually:
        <s1_root>/<city>/<product_id>/selected_s1_item.json

    So the product directory is the parent of selected_s1_item.json.
    """
    parent = selected_json_path.parent

    if parent == city_root:
        return parent

    return parent


def build_row_from_selected_json(
    city: str,
    selected_json_path: Path,
    city_root: Path,
    safe_root: Path,
) -> Dict[str, Any]:
    data = read_json(selected_json_path)

    properties = data.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    stac_id = first_non_empty(data.get("id"), properties.get("id"))

    product_id_candidates = [
        stac_id,
        properties.get("s1:product_identifier"),
        properties.get("safe:product_id"),
        properties.get("product_id"),
        properties.get("title"),
        selected_json_path.parent.name,
    ]

    product_id = clean_product_id(first_non_empty(*product_id_candidates))

    safe_name = safe_name_from_product_id(product_id)
    safe_zip_name = safe_zip_name_from_product_id(product_id)

    target_city_safe_root = safe_root / city
    target_safe_dir = target_city_safe_root / safe_name
    target_safe_zip = target_city_safe_root / safe_zip_name

    raw_product_dir = infer_city_product_dir(selected_json_path, city_root)
    raw_files = find_raw_files(raw_product_dir)

    asset_hrefs = extract_asset_hrefs(data)
    parsed_name = parse_s1_product_id(product_id)

    safe_dir_exists = target_safe_dir.exists() and target_safe_dir.is_dir()
    safe_zip_exists = target_safe_zip.exists() and target_safe_zip.is_file()

    is_pilot_city = city in PILOT_CITIES

    if safe_dir_exists:
        download_status = "SAFE_DIR_ALREADY_EXISTS"
    elif safe_zip_exists:
        download_status = "SAFE_ZIP_ALREADY_EXISTS"
    else:
        download_status = "NEEDS_DOWNLOAD"

    priority = 1 if is_pilot_city else 2

    datetime_value = first_non_empty(
        properties.get("datetime"),
        properties.get("start_datetime"),
        parsed_name.get("acquisition_start_from_name"),
    )

    row: Dict[str, Any] = {
        "city": city,
        "is_pilot_city": is_pilot_city,
        "download_priority": priority,
        "download_status": download_status,
        "needs_download": download_status == "NEEDS_DOWNLOAD",
        "selected_s1_item_json": str(selected_json_path),
        "raw_product_dir": str(raw_product_dir),
        "raw_vv_path": raw_files["raw_vv_path"],
        "raw_vh_path": raw_files["raw_vh_path"],
        "has_raw_vv": raw_files["has_raw_vv"],
        "has_raw_vh": raw_files["has_raw_vh"],
        "stac_id": stac_id,
        "safe_product_id": product_id,
        "safe_name": safe_name,
        "safe_zip_name": safe_zip_name,
        "target_safe_dir": str(target_safe_dir),
        "target_safe_zip": str(target_safe_zip),
        "safe_dir_exists": safe_dir_exists,
        "safe_zip_exists": safe_zip_exists,
        "datetime": parse_product_datetime(datetime_value),
        "start_datetime": parse_product_datetime(
            first_non_empty(properties.get("start_datetime"), parsed_name.get("acquisition_start_from_name"))
        ),
        "end_datetime": parse_product_datetime(
            first_non_empty(properties.get("end_datetime"), parsed_name.get("acquisition_stop_from_name"))
        ),
        "platform": first_non_empty(
            properties.get("platform"),
            properties.get("sat:platform_international_designator"),
            parsed_name.get("platform_from_name"),
        ),
        "constellation": first_non_empty(properties.get("constellation")),
        "orbit_state": first_non_empty(
            properties.get("sat:orbit_state"),
            properties.get("s1:orbit_state"),
        ),
        "relative_orbit": first_non_empty(
            properties.get("sat:relative_orbit"),
            properties.get("s1:relative_orbit"),
        ),
        "absolute_orbit": first_non_empty(
            properties.get("sat:absolute_orbit"),
            parsed_name.get("absolute_orbit_from_name"),
        ),
        "instrument_mode": first_non_empty(
            properties.get("sar:instrument_mode"),
            properties.get("s1:instrument_mode"),
            parsed_name.get("acquisition_mode_from_name"),
        ),
        "sar_product_type": first_non_empty(
            properties.get("sar:product_type"),
            properties.get("s1:product_type"),
            parsed_name.get("product_type_from_name"),
        ),
        "polarizations": first_non_empty(
            ",".join(properties.get("sar:polarizations", []))
            if isinstance(properties.get("sar:polarizations"), list)
            else properties.get("sar:polarizations"),
            parsed_name.get("polarization_code_from_name"),
        ),
        "resolution_range": first_non_empty(properties.get("sar:resolution_range")),
        "resolution_azimuth": first_non_empty(properties.get("sar:resolution_azimuth")),
        "asset_keys": " | ".join(sorted(asset_hrefs.keys())),
        "asset_href_vv": first_non_empty(
            asset_hrefs.get("vv"),
            asset_hrefs.get("VV"),
            asset_hrefs.get("measurement-vv"),
        ),
        "asset_href_vh": first_non_empty(
            asset_hrefs.get("vh"),
            asset_hrefs.get("VH"),
            asset_hrefs.get("measurement-vh"),
        ),
        "manual_search_name": safe_name,
        "cdse_odata_name_filter": f"Name eq '{safe_name}'",
        "asf_search_product_id": product_id,
        "copernicus_browser_search_text": product_id,
        "notes": (
            "Download the full SAFE product, not only VV/VH measurement GeoTIFFs. "
            "Store as target_safe_dir or target_safe_zip."
        ),
    }

    row.update(parsed_name)

    return row


def build_manifest(
    s1_root: Path,
    safe_root: Path,
    cities: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for city in tqdm(cities, desc="Preparing SAFE download manifest"):
        city = normalize_city_name(city)
        city_root = s1_root / city

        if not city_root.exists():
            rows.append(
                {
                    "city": city,
                    "download_status": "MISSING_CITY_S1_RAW_ROOT",
                    "needs_download": True,
                    "selected_s1_item_json": "",
                    "raw_product_dir": "",
                    "safe_product_id": "",
                    "safe_name": "",
                    "target_safe_dir": "",
                    "target_safe_zip": "",
                    "notes": f"City S1 raw root not found: {city_root}",
                }
            )
            continue

        selected_jsons = sorted(city_root.rglob("selected_s1_item.json"))

        if not selected_jsons:
            rows.append(
                {
                    "city": city,
                    "download_status": "MISSING_SELECTED_S1_ITEM_JSON",
                    "needs_download": True,
                    "selected_s1_item_json": "",
                    "raw_product_dir": "",
                    "safe_product_id": "",
                    "safe_name": "",
                    "target_safe_dir": "",
                    "target_safe_zip": "",
                    "notes": f"No selected_s1_item.json found under: {city_root}",
                }
            )
            continue

        for selected_json_path in selected_jsons:
            try:
                row = build_row_from_selected_json(
                    city=city,
                    selected_json_path=selected_json_path,
                    city_root=city_root,
                    safe_root=safe_root,
                )
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "city": city,
                        "download_status": "FAILED_READING_SELECTED_JSON",
                        "needs_download": True,
                        "selected_s1_item_json": str(selected_json_path),
                        "raw_product_dir": str(selected_json_path.parent),
                        "safe_product_id": "",
                        "safe_name": "",
                        "target_safe_dir": "",
                        "target_safe_zip": "",
                        "error": repr(exc),
                    }
                )

    return rows


def deduplicate_manifest_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate by city + safe_product_id.

    If duplicates exist, keep the row with the most complete raw VV/VH information.
    """
    df = pd.DataFrame(rows)

    if df.empty:
        return rows

    if "safe_product_id" not in df.columns:
        return rows

    df["has_raw_pair_score"] = (
        df.get("has_raw_vv", False).astype(str).str.lower().isin(["true", "1"])
        .astype(int)
        +
        df.get("has_raw_vh", False).astype(str).str.lower().isin(["true", "1"])
        .astype(int)
    )

    df = df.sort_values(
        by=["city", "safe_product_id", "has_raw_pair_score"],
        ascending=[True, True, False],
    )

    df = df.drop_duplicates(subset=["city", "safe_product_id"], keep="first")
    df = df.drop(columns=["has_raw_pair_score"])

    return df.to_dict(orient="records")


def write_text_list(path: Path, product_names: Sequence[str]) -> None:
    ensure_dir(path.parent)

    unique_names = sorted(set(name for name in product_names if isinstance(name, str) and name))

    path.write_text("\n".join(unique_names) + ("\n" if unique_names else ""), encoding="utf-8")


def write_outputs(output_root: Path, rows: List[Dict[str, Any]]) -> None:
    metadata_root = output_root / "metadata"
    qc_root = output_root / "qc"

    ensure_dir(metadata_root)
    ensure_dir(qc_root)

    manifest_csv = metadata_root / "s1_safe_download_manifest.csv"
    pilot_manifest_csv = metadata_root / "s1_safe_download_manifest_pilot.csv"
    product_names_txt = metadata_root / "s1_safe_product_names_to_download.txt"
    pilot_product_names_txt = metadata_root / "s1_safe_product_names_to_download_pilot.txt"
    summary_csv = qc_root / "s1_safe_download_manifest_summary.csv"

    df = pd.DataFrame(rows)

    if not df.empty:
        preferred_columns = [
            "city",
            "is_pilot_city",
            "download_priority",
            "download_status",
            "needs_download",
            "safe_product_id",
            "safe_name",
            "safe_zip_name",
            "target_safe_dir",
            "target_safe_zip",
            "safe_dir_exists",
            "safe_zip_exists",
            "datetime",
            "start_datetime",
            "end_datetime",
            "platform",
            "constellation",
            "orbit_state",
            "relative_orbit",
            "absolute_orbit",
            "instrument_mode",
            "sar_product_type",
            "polarizations",
            "selected_s1_item_json",
            "raw_product_dir",
            "raw_vv_path",
            "raw_vh_path",
            "has_raw_vv",
            "has_raw_vh",
            "asset_keys",
            "asset_href_vv",
            "asset_href_vh",
            "manual_search_name",
            "cdse_odata_name_filter",
            "asf_search_product_id",
            "copernicus_browser_search_text",
            "notes",
        ]

        existing_preferred = [col for col in preferred_columns if col in df.columns]
        remaining = [col for col in df.columns if col not in existing_preferred]

        df = df[existing_preferred + remaining]

    df.to_csv(manifest_csv, index=False)

    pilot_df = df[df["city"].isin(PILOT_CITIES)] if "city" in df.columns else pd.DataFrame()
    pilot_df.to_csv(pilot_manifest_csv, index=False)

    if "needs_download" in df.columns and "safe_name" in df.columns:
        names_to_download = df.loc[df["needs_download"] == True, "safe_name"].dropna().astype(str).tolist()
    else:
        names_to_download = []

    if not pilot_df.empty and "needs_download" in pilot_df.columns and "safe_name" in pilot_df.columns:
        pilot_names_to_download = (
            pilot_df.loc[pilot_df["needs_download"] == True, "safe_name"]
            .dropna()
            .astype(str)
            .tolist()
        )
    else:
        pilot_names_to_download = []

    write_text_list(product_names_txt, names_to_download)
    write_text_list(pilot_product_names_txt, pilot_names_to_download)

    if "download_status" in df.columns:
        summary_df = (
            df["download_status"]
            .value_counts(dropna=False)
            .rename_axis("download_status")
            .reset_index(name="product_count")
        )
    else:
        summary_df = pd.DataFrame(columns=["download_status", "product_count"])

    summary_df.to_csv(summary_csv, index=False)

    print(f"[INFO] Wrote manifest: {manifest_csv}")
    print(f"[INFO] Wrote pilot manifest: {pilot_manifest_csv}")
    print(f"[INFO] Wrote product list: {product_names_txt}")
    print(f"[INFO] Wrote pilot product list: {pilot_product_names_txt}")
    print(f"[INFO] Wrote summary: {summary_csv}")

    print("[INFO] Download status counts:")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No rows")

    if not pilot_df.empty:
        print("[INFO] Pilot products to download:")
        pilot_cols = [
            "city",
            "download_status",
            "safe_name",
            "target_safe_dir",
            "target_safe_zip",
        ]
        pilot_cols = [col for col in pilot_cols if col in pilot_df.columns]
        print(pilot_df[pilot_cols].to_string(index=False))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    input_root = Path(str(cfg["input_root"]))
    output_root = Path(str(cfg["output_root"]))
    s1_root = Path(str(cfg["s1_root"]))

    safe_root = args.safe_root if args.safe_root is not None else input_root / "s1_images" / "safe"

    cities = discover_cities(
    output_root=output_root,
    s1_root=s1_root,
    selected_cities=args.city,
    pilot_only=args.pilot_only,)

    print("[INFO] Sentinel-1 SAFE download manifest preparation")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Input root: {input_root}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] S1 raw root: {s1_root}")
    print(f"[INFO] Target SAFE root: {safe_root}")
    print(f"[INFO] Cities selected: {len(cities)}")

    rows = build_manifest(
        s1_root=s1_root,
        safe_root=safe_root,
        cities=cities,
    )

    rows = deduplicate_manifest_rows(rows)

    write_outputs(output_root=output_root, rows=rows)

    df = pd.DataFrame(rows)

    if not df.empty and "needs_download" in df.columns:
        needs_download_count = int((df["needs_download"] == True).sum())
        print(f"[INFO] Products needing download: {needs_download_count}")

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())