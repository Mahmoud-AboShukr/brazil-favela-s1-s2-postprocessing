#!/usr/bin/env python3
"""
Search fallback Sentinel-1 GRD products for missing S1 coverage AOIs.

This script reads WGS84 AOI GeoJSON files created by:

    09_extract_s1_missing_coverage_aoi.py

Then it searches a STAC catalog, by default Microsoft Planetary Computer:

    https://planetarycomputer.microsoft.com/api/stac/v1
    collection: sentinel-1-grd

For each city, it ranks Sentinel-1 GRD candidates by overlap with the missing
S1 nodata AOI.

Important:
- This script does not download or preprocess data.
- This script does not modify rasters.
- Its purpose is to produce candidate tables for manual/product selection.
- For SNAP-based repair, the selected item ID can later be used to locate/download
  the corresponding SAFE product from Copernicus Data Space / ASF / another archive.

Outputs:
    qc/s1_fallback_search/
        per_city/<city>_s1_fallback_candidates.csv
        per_city/<city>_s1_fallback_candidates.json
        combined_s1_fallback_candidates.csv
        combined_s1_fallback_candidates.json
        combined_s1_fallback_candidates.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union


DEFAULT_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEFAULT_COLLECTION = "sentinel-1-grd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search fallback Sentinel-1 GRD products for extracted S1 missing AOIs."
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
        "--aoi-dir",
        type=str,
        default=None,
        help=(
            "Directory containing WGS84 AOI GeoJSONs. If omitted, uses "
            "<instance-root>/qc/s1_missing_aoi/wgs84"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. If omitted, uses "
            "<instance-root>/qc/s1_fallback_search"
        ),
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help=(
            "Cities to search. If omitted, inferred from *_s1_missing_aoi_wgs84.geojson "
            "files in --aoi-dir."
        ),
    )

    parser.add_argument(
        "--stac-url",
        type=str,
        default=DEFAULT_STAC_URL,
        help=f"STAC API URL. Default: {DEFAULT_STAC_URL}",
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"STAC collection ID. Default: {DEFAULT_COLLECTION}",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="2022-01-01",
        help="Search start date in YYYY-MM-DD format. Default: 2022-01-01",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default="2022-12-31",
        help="Search end date in YYYY-MM-DD format. Default: 2022-12-31",
    )

    parser.add_argument(
        "--instrument-mode",
        type=str,
        default="IW",
        help="Preferred/required S1 instrument mode. Default: IW. Use 'ANY' to disable.",
    )

    parser.add_argument(
        "--required-polarizations",
        nargs="*",
        default=["VV", "VH"],
        help="Required polarizations. Default: VV VH. Use empty string to disable.",
    )

    parser.add_argument(
        "--orbit-state",
        choices=["ASCENDING", "DESCENDING", "ANY"],
        default="ANY",
        help="Optional orbit direction filter. Default: ANY.",
    )

    parser.add_argument(
        "--min-overlap-percent",
        type=float,
        default=1.0,
        help="Minimum overlap with missing AOI, as percent of AOI area. Default: 1.0",
    )

    parser.add_argument(
        "--full-coverage-threshold-percent",
        type=float,
        default=95.0,
        help="Candidate considered full coverage if overlap >= this percent. Default: 95.0",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=300,
        help="Maximum STAC items to evaluate per city. Default: 300",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top candidates to keep per city. Default: 20",
    )

    parser.add_argument(
        "--ranking-date",
        type=str,
        default=None,
        help=(
            "Optional date used to rank temporal closeness, YYYY-MM-DD. "
            "If omitted, midpoint of start/end date is used."
        ),
    )

    parser.add_argument(
        "--sign-assets",
        action="store_true",
        help=(
            "Attempt to sign Planetary Computer asset URLs using planetary_computer. "
            "Usually not needed for candidate search."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser.parse_args()


def require_pystac_client():
    try:
        from pystac_client import Client
    except ImportError as exc:
        raise ImportError(
            "This script requires pystac-client. Install it with:\n"
            "  pip install pystac-client\n"
        ) from exc

    return Client


def maybe_sign_item(item, sign_assets: bool):
    if not sign_assets:
        return item

    try:
        import planetary_computer
    except ImportError:
        print(
            "[WARN] --sign-assets requested but planetary_computer is not installed. "
            "Continuing without signed asset URLs.",
            file=sys.stderr,
        )
        return item

    try:
        return planetary_computer.sign(item)
    except Exception as exc:
        print(f"[WARN] Could not sign item {item.id}: {exc}", file=sys.stderr)
        return item


def parse_date(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)


def midpoint_date(start_date: str, end_date: str) -> datetime:
    start = parse_date(start_date)
    end = parse_date(end_date)
    return start + (end - start) / 2


def safe_remove(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")


def discover_cities(aoi_dir: Path, cities: list[str] | None) -> list[str]:
    if cities:
        return sorted(cities)

    discovered = []

    for path in sorted(aoi_dir.glob("*_s1_missing_aoi_wgs84.geojson")):
        city = path.name.replace("_s1_missing_aoi_wgs84.geojson", "")
        discovered.append(city)

    if not discovered:
        raise FileNotFoundError(f"No AOI GeoJSON files found under {aoi_dir}")

    return sorted(discovered)


def find_aoi_geojson(aoi_dir: Path, city: str) -> Path:
    path = aoi_dir / f"{city}_s1_missing_aoi_wgs84.geojson"

    if not path.exists():
        raise FileNotFoundError(f"AOI GeoJSON not found for {city}: {path}")

    return path


def read_aoi_geometry_wgs84(path: Path):
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"AOI file is empty: {path}")

    if gdf.crs is None:
        print(f"[WARN] AOI has no CRS; assuming EPSG:4326: {path}", file=sys.stderr)
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    geom = unary_union(gdf.geometry)

    if geom.is_empty:
        raise ValueError(f"AOI geometry is empty: {path}")

    return geom


def estimate_equal_area_crs() -> CRS:
    """
    Use a global equal-area CRS for overlap areas.

    EPSG:6933 = WGS 84 / NSIDC EASE-Grid 2.0 Global, suitable for global
    equal-area calculations.
    """
    return CRS.from_epsg(6933)


def area_in_equal_area(geom_wgs84, equal_area_crs: CRS) -> float:
    gdf = gpd.GeoDataFrame([{"geometry": geom_wgs84}], crs="EPSG:4326")
    projected = gdf.to_crs(equal_area_crs)
    return float(projected.geometry.iloc[0].area)


def overlap_stats(aoi_wgs84, item_geom_wgs84, equal_area_crs: CRS) -> dict:
    gdf = gpd.GeoDataFrame(
        [
            {"kind": "aoi", "geometry": aoi_wgs84},
            {"kind": "item", "geometry": item_geom_wgs84},
        ],
        crs="EPSG:4326",
    ).to_crs(equal_area_crs)

    aoi_geom = gdf.geometry.iloc[0]
    item_geom = gdf.geometry.iloc[1]

    aoi_area = float(aoi_geom.area)
    item_area = float(item_geom.area)
    intersection = aoi_geom.intersection(item_geom)
    overlap_area = float(intersection.area)

    return {
        "aoi_area_m2": aoi_area,
        "item_area_m2": item_area,
        "overlap_area_m2": overlap_area,
        "overlap_percent_of_missing_aoi": 100.0 * overlap_area / aoi_area if aoi_area > 0 else 0.0,
        "overlap_percent_of_item": 100.0 * overlap_area / item_area if item_area > 0 else 0.0,
    }


def get_item_geometry(item):
    if getattr(item, "geometry", None):
        try:
            return shape(item.geometry)
        except Exception:
            pass

    if getattr(item, "bbox", None):
        return box(*item.bbox)

    raise ValueError(f"Item has no usable geometry/bbox: {item.id}")


def normalize_polarizations(value: Any) -> set[str]:
    if value is None:
        return set()

    if isinstance(value, str):
        parts = value.replace(",", " ").replace("/", " ").split()
        return {p.upper() for p in parts}

    if isinstance(value, list):
        return {str(v).upper() for v in value}

    return {str(value).upper()}


def item_polarizations(item) -> set[str]:
    props = item.properties or {}

    for key in [
        "sar:polarizations",
        "s1:polarizations",
        "s1:polarization",
        "polarization",
    ]:
        pols = normalize_polarizations(props.get(key))
        if pols:
            return pols

    # Fallback from asset keys
    keys = {k.upper() for k in item.assets.keys()}
    pols = set()
    if "VV" in keys:
        pols.add("VV")
    if "VH" in keys:
        pols.add("VH")
    if "HH" in keys:
        pols.add("HH")
    if "HV" in keys:
        pols.add("HV")
    return pols


def item_instrument_mode(item) -> str:
    props = item.properties or {}

    for key in [
        "sar:instrument_mode",
        "s1:instrument_mode",
        "instrument_mode",
    ]:
        value = props.get(key)
        if value:
            return str(value).upper()

    # Fallback from Sentinel-1 product ID, e.g. S1A_IW_GRDH...
    parts = item.id.split("_")
    if len(parts) > 1:
        return parts[1].upper()

    return ""


def item_product_type(item) -> str:
    props = item.properties or {}

    for key in [
        "s1:product_type",
        "sar:product_type",
        "product_type",
    ]:
        value = props.get(key)
        if value:
            return str(value).upper()

    # Fallback from item ID, e.g. S1A_IW_GRDH...
    parts = item.id.split("_")
    if len(parts) > 2:
        return parts[2].upper()

    return ""


def item_orbit_state(item) -> str:
    props = item.properties or {}

    for key in [
        "sat:orbit_state",
        "s1:orbit_state",
        "orbit_state",
    ]:
        value = props.get(key)
        if value:
            return str(value).upper()

    return ""


def get_asset_href(item, preferred_keys: list[str]) -> str:
    for key in preferred_keys:
        if key in item.assets:
            return item.assets[key].href

    lower_map = {k.lower(): k for k in item.assets.keys()}

    for key in preferred_keys:
        lower = key.lower()
        if lower in lower_map:
            return item.assets[lower_map[lower]].href

    return ""


def get_item_datetime(item) -> datetime | None:
    dt = item.datetime

    if dt is not None:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    value = item.properties.get("datetime") if item.properties else None
    if value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None

    return None


def temporal_distance_days(item_dt: datetime | None, ranking_date: datetime) -> float:
    if item_dt is None:
        return math.inf

    return abs((item_dt - ranking_date).total_seconds()) / 86400.0


def item_passes_filters(
    item,
    required_polarizations: list[str],
    instrument_mode: str,
    orbit_state: str,
) -> tuple[bool, str]:
    pols = item_polarizations(item)
    mode = item_instrument_mode(item)
    product_type = item_product_type(item)
    item_orbit = item_orbit_state(item)

    required = {p.upper() for p in required_polarizations if p.strip()}

    if required and not required.issubset(pols):
        return False, f"missing required polarizations {sorted(required - pols)}"

    if instrument_mode.upper() != "ANY":
        if mode and mode != instrument_mode.upper():
            return False, f"instrument mode {mode} != {instrument_mode.upper()}"
        if not mode and f"_{instrument_mode.upper()}_" not in item.id.upper():
            return False, f"could not confirm instrument mode {instrument_mode.upper()}"

    # We want GRD/GRDH products.
    if product_type and "GRD" not in product_type:
        return False, f"product type {product_type} does not look like GRD"

    if orbit_state.upper() != "ANY":
        if item_orbit and item_orbit != orbit_state.upper():
            return False, f"orbit state {item_orbit} != {orbit_state.upper()}"

    return True, "passed"


def build_candidate_record(
    city: str,
    item,
    aoi_geom_wgs84,
    equal_area_crs: CRS,
    ranking_date: datetime,
    full_coverage_threshold_percent: float,
) -> dict:
    item_geom = get_item_geometry(item)
    overlap = overlap_stats(
        aoi_wgs84=aoi_geom_wgs84,
        item_geom_wgs84=item_geom,
        equal_area_crs=equal_area_crs,
    )

    item_dt = get_item_datetime(item)
    dt_text = item_dt.isoformat() if item_dt else ""

    props = item.properties or {}
    pols = sorted(item_polarizations(item))
    mode = item_instrument_mode(item)
    product_type = item_product_type(item)
    orbit_state = item_orbit_state(item)

    vv_href = get_asset_href(item, ["vv", "VV"])
    vh_href = get_asset_href(item, ["vh", "VH"])
    manifest_href = get_asset_href(item, ["manifest", "safe-manifest", "manifest.safe"])
    preview_href = get_asset_href(item, ["rendered_preview", "preview", "thumbnail"])

    overlap_pct = overlap["overlap_percent_of_missing_aoi"]
    full_coverage = overlap_pct >= full_coverage_threshold_percent

    record = {
        "city": city,
        "item_id": item.id,
        "datetime": dt_text,
        "platform": props.get("platform", ""),
        "constellation": props.get("constellation", ""),
        "instrument_mode": mode,
        "product_type": product_type,
        "orbit_state": orbit_state,
        "relative_orbit": props.get("sat:relative_orbit", props.get("s1:relative_orbit", "")),
        "absolute_orbit": props.get("sat:absolute_orbit", props.get("s1:absolute_orbit", "")),
        "polarizations": ",".join(pols),
        "asset_keys": ",".join(sorted(item.assets.keys())),
        "has_vv_asset": bool(vv_href),
        "has_vh_asset": bool(vh_href),
        "vv_asset_href": vv_href,
        "vh_asset_href": vh_href,
        "manifest_asset_href": manifest_href,
        "preview_href": preview_href,
        "aoi_area_m2": overlap["aoi_area_m2"],
        "item_area_m2": overlap["item_area_m2"],
        "overlap_area_m2": overlap["overlap_area_m2"],
        "overlap_percent_of_missing_aoi": overlap_pct,
        "overlap_percent_of_item": overlap["overlap_percent_of_item"],
        "full_coverage_candidate": full_coverage,
        "temporal_distance_days_from_ranking_date": temporal_distance_days(item_dt, ranking_date),
        "ranking_date": ranking_date.date().isoformat(),
        "item_bbox": ",".join(str(v) for v in item.bbox) if item.bbox else "",
        "stac_collection": item.collection_id,
        "candidate_note": (
            "Use item_id to locate/download corresponding SAFE from Copernicus/ASF "
            "if SNAP SAFE preprocessing is required."
        ),
    }

    return record


def rank_candidates(records: list[dict]) -> list[dict]:
    ranked = sorted(
        records,
        key=lambda r: (
            -float(r["overlap_percent_of_missing_aoi"]),
            float(r["temporal_distance_days_from_ranking_date"])
            if r["temporal_distance_days_from_ranking_date"] != math.inf
            else 999999.0,
            str(r["item_id"]),
        ),
    )

    for i, row in enumerate(ranked, start=1):
        row["rank"] = i

    return ranked


def search_city(
    client,
    city: str,
    aoi_path: Path,
    args: argparse.Namespace,
    ranking_date: datetime,
) -> list[dict]:
    aoi_geom = read_aoi_geometry_wgs84(aoi_path)
    equal_area_crs = estimate_equal_area_crs()

    datetime_range = f"{args.start_date}/{args.end_date}"

    print(f"[INFO] Searching STAC for {city}")
    print(f"[INFO] AOI: {aoi_path}")
    print(f"[INFO] datetime: {datetime_range}")

    search = client.search(
        collections=[args.collection],
        intersects=mapping(aoi_geom),
        datetime=datetime_range,
        max_items=args.max_items,
    )

    raw_items = []
    for i, item in enumerate(search.items()):
        if i >= args.max_items:
            break
        raw_items.append(maybe_sign_item(item, args.sign_assets))

    print(f"[INFO] Raw STAC items returned: {len(raw_items)}")

    records = []
    rejected = 0

    for item in raw_items:
        passed, reason = item_passes_filters(
            item=item,
            required_polarizations=args.required_polarizations,
            instrument_mode=args.instrument_mode,
            orbit_state=args.orbit_state,
        )

        if not passed:
            rejected += 1
            continue

        try:
            record = build_candidate_record(
                city=city,
                item=item,
                aoi_geom_wgs84=aoi_geom,
                equal_area_crs=equal_area_crs,
                ranking_date=ranking_date,
                full_coverage_threshold_percent=args.full_coverage_threshold_percent,
            )
        except Exception as exc:
            print(f"[WARN] Failed to build candidate for {item.id}: {exc}", file=sys.stderr)
            rejected += 1
            continue

        if record["overlap_percent_of_missing_aoi"] < args.min_overlap_percent:
            rejected += 1
            continue

        records.append(record)

    ranked = rank_candidates(records)
    top = ranked[: args.top_n]

    print(f"[INFO] Candidates after filters: {len(ranked)}")
    print(f"[INFO] Rejected/filtered items: {rejected}")

    if top:
        best = top[0]
        print(
            "[INFO] Best candidate: "
            f"{best['item_id']} | overlap={best['overlap_percent_of_missing_aoi']:.2f}% | "
            f"date={best['datetime']} | orbit={best['orbit_state']}"
        )
    else:
        print("[WARN] No candidates found after filters.")

    return top


def write_csv(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    if not rows:
        # Write an empty marker file with minimal fields.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["no_candidates"])
        return

    fields = []
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
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def write_markdown(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "city",
        "rank",
        "item_id",
        "datetime",
        "overlap_percent_of_missing_aoi",
        "full_coverage_candidate",
        "orbit_state",
        "relative_orbit",
        "polarizations",
        "instrument_mode",
        "product_type",
        "temporal_distance_days_from_ranking_date",
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Fallback Sentinel-1 GRD candidate search\n\n")
        f.write(
            "This report ranks Sentinel-1 GRD products by overlap with extracted "
            "S1 missing-coverage AOIs. The products are candidates for a later "
            "download + SNAP preprocessing + missing-pixel fill workflow.\n\n"
        )

        if not rows:
            f.write("No candidates found.\n")
            return

        f.write("## Candidate counts by city\n\n")
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["city"]] = counts.get(row["city"], 0) + 1

        for city, count in sorted(counts.items()):
            f.write(f"- `{city}`: {count} candidates\n")

        f.write("\n## Top candidates\n\n")
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

    Client = require_pystac_client()

    instance_root = Path(args.instance_root)
    aoi_dir = Path(args.aoi_dir) if args.aoi_dir else instance_root / "qc" / "s1_missing_aoi" / "wgs84"
    output_dir = Path(args.output_dir) if args.output_dir else instance_root / "qc" / "s1_fallback_search"
    per_city_dir = output_dir / "per_city"

    if not instance_root.exists():
        raise FileNotFoundError(f"Instance root does not exist: {instance_root}")

    if not aoi_dir.exists():
        raise FileNotFoundError(f"AOI directory does not exist: {aoi_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_city_dir.mkdir(parents=True, exist_ok=True)

    cities = discover_cities(aoi_dir, args.cities)

    if args.ranking_date:
        ranking_date = parse_date(args.ranking_date)
    else:
        ranking_date = midpoint_date(args.start_date, args.end_date)

    print(f"[INFO] Instance root: {instance_root}")
    print(f"[INFO] AOI dir: {aoi_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] STAC URL: {args.stac_url}")
    print(f"[INFO] Collection: {args.collection}")
    print(f"[INFO] Date range: {args.start_date}/{args.end_date}")
    print(f"[INFO] Ranking date: {ranking_date.date().isoformat()}")
    print(f"[INFO] Cities: {cities}")
    print(f"[INFO] Required polarizations: {args.required_polarizations}")
    print(f"[INFO] Instrument mode: {args.instrument_mode}")
    print(f"[INFO] Orbit state: {args.orbit_state}")
    print(f"[INFO] min_overlap_percent: {args.min_overlap_percent}")
    print(f"[INFO] max_items: {args.max_items}")
    print(f"[INFO] top_n: {args.top_n}")

    print("[INFO] Opening STAC client...")
    client = Client.open(args.stac_url)

    all_records = []

    for i, city in enumerate(cities, start=1):
        print(f"\n[STEP {i}/{len(cities)}] {city}")

        try:
            aoi_path = find_aoi_geojson(aoi_dir, city)
            records = search_city(
                client=client,
                city=city,
                aoi_path=aoi_path,
                args=args,
                ranking_date=ranking_date,
            )
            all_records.extend(records)

            city_csv = per_city_dir / f"{city}_s1_fallback_candidates.csv"
            city_json = per_city_dir / f"{city}_s1_fallback_candidates.json"

            write_csv(records, city_csv, overwrite=args.overwrite)
            write_json(records, city_json, overwrite=args.overwrite)

            print(f"[OK] Wrote per-city candidates: {city_csv}")

        except Exception as exc:
            print(f"[ERROR] {city}: {exc}", file=sys.stderr)

            error_record = {
                "city": city,
                "rank": "",
                "item_id": "",
                "error": str(exc),
            }
            all_records.append(error_record)

    combined_csv = output_dir / "combined_s1_fallback_candidates.csv"
    combined_json = output_dir / "combined_s1_fallback_candidates.json"
    combined_md = output_dir / "combined_s1_fallback_candidates.md"

    write_csv(all_records, combined_csv, overwrite=args.overwrite)
    write_json(all_records, combined_json, overwrite=args.overwrite)
    write_markdown(all_records, combined_md, overwrite=args.overwrite)

    print("\n[DONE] Wrote:")
    print(f"  CSV:  {combined_csv}")
    print(f"  JSON: {combined_json}")
    print(f"  MD:   {combined_md}")
    print(f"  Per-city outputs: {per_city_dir}")

    print("\n[SUMMARY]")
    counts: dict[str, int] = {}
    for row in all_records:
        city = row.get("city", "unknown")
        if row.get("item_id"):
            counts[city] = counts.get(city, 0) + 1
        else:
            counts[city] = counts.get(city, 0)

    for city, count in sorted(counts.items()):
        print(f"  {city}: {count} candidates")


if __name__ == "__main__":
    main()