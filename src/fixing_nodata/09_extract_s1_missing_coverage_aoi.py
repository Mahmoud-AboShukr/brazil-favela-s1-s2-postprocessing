#!/usr/bin/env python3
"""
Extract S1 missing-coverage AOI polygons.

This script extracts vector polygons from S1 nodata masks in the repaired instance.
It is intended for cities where S1 has large footprint/coverage gaps, especially:

    campo_grande
    sao_goncalo

It does not modify rasters.

Outputs:
    qc/s1_missing_aoi/
        native/<city>_s1_missing_aoi_native.gpkg
        wgs84/<city>_s1_missing_aoi_wgs84.geojson
        all_s1_missing_aois_native.gpkg
        all_s1_missing_aois_wgs84.geojson
        s1_missing_aoi_summary.csv/json/md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract S1 missing-coverage AOIs from S1 nodata masks."
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
        "--s1-subdir",
        type=str,
        default="s1_snap",
        help="S1 subdirectory inside instance root. Default: s1_snap",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. If omitted, outputs are written to "
            "<instance-root>/qc/s1_missing_aoi"
        ),
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help=(
            "Cities to process. If omitted, all cities under S1 root are processed. "
            "Recommended first run: campo_grande sao_goncalo sorocaba"
        ),
    )

    s1_zero_group = parser.add_mutually_exclusive_group()
    s1_zero_group.add_argument(
        "--s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_true",
        help="Treat all-zero-all-band S1 pixels as nodata.",
    )
    s1_zero_group.add_argument(
        "--no-s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_false",
        help="Do not treat all-zero-all-band S1 pixels as nodata.",
    )
    parser.set_defaults(s1_all_zero_as_nodata=False)

    nan_group = parser.add_mutually_exclusive_group()
    nan_group.add_argument(
        "--nan-as-nodata",
        dest="nan_as_nodata",
        action="store_true",
        help="Treat NaN/Inf values in any S1 band as nodata.",
    )
    nan_group.add_argument(
        "--no-nan-as-nodata",
        dest="nan_as_nodata",
        action="store_false",
        help="Do not treat NaN/Inf values as nodata.",
    )
    parser.set_defaults(nan_as_nodata=True)

    parser.add_argument(
        "--min-area-pixels",
        type=int,
        default=1,
        help=(
            "Minimum connected polygon area in pixels to keep before dissolve. "
            "Use 1 for exact extraction. Increase to remove tiny specks."
        ),
    )

    parser.add_argument(
        "--buffer-meters",
        type=float,
        default=500.0,
        help=(
            "Optional buffer in native projected units before WGS84 export. "
            "Useful for product search AOIs. Default: 500 m."
        ),
    )

    parser.add_argument(
        "--simplify-tolerance-meters",
        type=float,
        default=30.0,
        help=(
            "Simplification tolerance in native projected units after dissolve/buffer. "
            "Use 0 to disable. Default: 30 m."
        ),
    )

    parser.add_argument(
        "--write-empty-records",
        action="store_true",
        help="Include no-nodata cities in summary with no geometry outputs.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser.parse_args()


def require_geopandas():
    try:
        import geopandas as gpd
        from pyproj import CRS
    except ImportError as exc:
        raise ImportError(
            "This script requires geopandas and pyproj. Install them with:\n"
            "  pip install geopandas pyproj shapely fiona\n"
            "or use your existing geospatial environment."
        ) from exc

    return gpd, CRS


def percent(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def discover_cities(s1_root: Path, cities: list[str] | None) -> list[str]:
    if cities:
        return sorted(cities)

    discovered = sorted([p.name for p in s1_root.iterdir() if p.is_dir()])

    if not discovered:
        raise FileNotFoundError(f"No city folders found under {s1_root}")

    return discovered


def find_s1_raster(s1_root: Path, city: str) -> Path:
    city_dir = s1_root / city

    candidates = sorted(city_dir.glob(f"{city}_s1_snap_vv_vh_vvdiff_10m_aligned.tif"))

    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s1*.tif"))

    if not candidates:
        raise FileNotFoundError(f"No S1 raster found for {city} under {city_dir}")

    return candidates[0]


def build_s1_nodata_mask(
    s1_path: Path,
    all_zero_as_nodata: bool,
    nan_as_nodata: bool,
) -> tuple[np.ndarray, dict]:
    with rasterio.open(s1_path) as src:
        height = src.height
        width = src.width
        count = src.count
        total_pixels = height * width

        combined = np.zeros((height, width), dtype=bool)
        official = np.zeros((height, width), dtype=bool)
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
            block_combined = official_block.copy()

            official[row0:row1, col0:col1] = official_block

            if all_zero_as_nodata or nan_as_nodata:
                data = src.read(indexes=indexes, window=window)

                if all_zero_as_nodata:
                    zero_block = np.all(data == 0, axis=0)
                    all_zero[row0:row1, col0:col1] = zero_block
                    block_combined |= zero_block

                if nan_as_nodata:
                    nonfinite_block = np.any(~np.isfinite(data), axis=0)
                    nonfinite[row0:row1, col0:col1] = nonfinite_block
                    block_combined |= nonfinite_block

            combined[row0:row1, col0:col1] = block_combined

        meta = {
            "width": width,
            "height": height,
            "band_count": count,
            "dtype": src.dtypes[0],
            "crs": src.crs,
            "transform": src.transform,
            "nodata_value": src.nodata,
            "total_pixels": int(total_pixels),
            "official_nodata_pixels": int(official.sum()),
            "all_zero_allbands_pixels": int(all_zero.sum()),
            "nonfinite_pixels": int(nonfinite.sum()),
            "combined_nodata_pixels": int(combined.sum()),
            "official_nodata_percent": percent(int(official.sum()), total_pixels),
            "all_zero_allbands_percent": percent(int(all_zero.sum()), total_pixels),
            "nonfinite_percent": percent(int(nonfinite.sum()), total_pixels),
            "combined_nodata_percent": percent(int(combined.sum()), total_pixels),
        }

    return combined, meta


def polygonize_nodata_mask(
    nodata_mask: np.ndarray,
    transform,
    pixel_area_native: float,
    min_area_pixels: int,
) -> list:
    mask_uint8 = nodata_mask.astype(np.uint8)

    polygons = []

    for geom, value in shapes(mask_uint8, mask=nodata_mask, transform=transform):
        if int(value) != 1:
            continue

        poly = shape(geom)

        if poly.is_empty:
            continue

        min_area_native = float(min_area_pixels) * float(pixel_area_native)

        if poly.area < min_area_native:
            continue

        polygons.append(poly)

    return polygons


def clean_geometry(geom):
    if geom.is_empty:
        return geom

    try:
        fixed = geom.buffer(0)
        if not fixed.is_empty:
            return fixed
    except Exception:
        pass

    return geom


def output_paths(output_dir: Path, city: str) -> dict:
    return {
        "native_city": output_dir / "native" / f"{city}_s1_missing_aoi_native.gpkg",
        "wgs84_city": output_dir / "wgs84" / f"{city}_s1_missing_aoi_wgs84.geojson",
    }


def safe_remove(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")


def write_city_outputs(
    gpd,
    city: str,
    geom_native,
    crs_native,
    row_attrs: dict,
    output_dir: Path,
    overwrite: bool,
) -> tuple[str, str, dict]:
    paths = output_paths(output_dir, city)

    paths["native_city"].parent.mkdir(parents=True, exist_ok=True)
    paths["wgs84_city"].parent.mkdir(parents=True, exist_ok=True)

    gdf_native = gpd.GeoDataFrame(
        [{**row_attrs, "geometry": geom_native}],
        crs=crs_native,
    )

    gdf_wgs84 = gdf_native.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    centroid = gdf_wgs84.geometry.iloc[0].centroid

    row_extra = {
        "bbox_wgs84_minx": float(minx),
        "bbox_wgs84_miny": float(miny),
        "bbox_wgs84_maxx": float(maxx),
        "bbox_wgs84_maxy": float(maxy),
        "centroid_lon": float(centroid.x),
        "centroid_lat": float(centroid.y),
        "native_output_path": str(paths["native_city"]),
        "wgs84_output_path": str(paths["wgs84_city"]),
    }

    safe_remove(paths["native_city"], overwrite=overwrite)
    safe_remove(paths["wgs84_city"], overwrite=overwrite)

    gdf_native.to_file(paths["native_city"], layer="s1_missing_aoi", driver="GPKG")
    gdf_wgs84.to_file(paths["wgs84_city"], driver="GeoJSON")

    return str(paths["native_city"]), str(paths["wgs84_city"]), row_extra


def write_aggregate_outputs(
    gpd,
    records: list[dict],
    output_dir: Path,
    overwrite: bool,
) -> None:
    geom_records = [r for r in records if r.get("geometry") is not None]

    if not geom_records:
        return

    gdf = gpd.GeoDataFrame(geom_records, geometry="geometry", crs=geom_records[0]["crs_native"])

    # If all CRS are same, this works directly. They should be same if all products
    # are on the same output CRS family, but we still keep per-city files as the source of truth.
    native_path = output_dir / "all_s1_missing_aois_native.gpkg"
    wgs84_path = output_dir / "all_s1_missing_aois_wgs84.geojson"

    safe_remove(native_path, overwrite=overwrite)
    safe_remove(wgs84_path, overwrite=overwrite)

    drop_crs = gdf.drop(columns=["crs_native"], errors="ignore")
    drop_crs.to_file(native_path, layer="s1_missing_aois", driver="GPKG")
    drop_crs.to_crs("EPSG:4326").to_file(wgs84_path, driver="GeoJSON")


def write_csv(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    if not rows:
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key == "geometry":
                continue
            if key == "crs_native":
                continue
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            safe_row = {k: v for k, v in row.items() if k in fields}
            writer.writerow(safe_row)


def write_json(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    safe_rows = []

    for row in rows:
        safe_row = {}
        for key, value in row.items():
            if key in {"geometry", "crs_native"}:
                continue
            safe_row[key] = value
        safe_rows.append(safe_row)

    with path.open("w", encoding="utf-8") as f:
        json.dump(safe_rows, f, indent=2, ensure_ascii=False)


def write_markdown(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    cols = [
        "city",
        "status",
        "s1_combined_nodata_percent",
        "missing_polygon_count_before_dissolve",
        "missing_area_native",
        "bbox_wgs84_minx",
        "bbox_wgs84_miny",
        "bbox_wgs84_maxx",
        "bbox_wgs84_maxy",
        "wgs84_output_path",
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# S1 missing-coverage AOI extraction\n\n")
        f.write(
            "This report lists the S1 nodata footprints extracted as polygons. "
            "Use the EPSG:4326 GeoJSON outputs as AOIs for fallback Sentinel-1 product search.\n\n"
        )

        f.write("## Status counts\n\n")
        counts = {}
        for row in rows:
            status = row.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1

        for status, count in sorted(counts.items()):
            f.write(f"- `{status}`: {count}\n")

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


def process_city(gpd, city: str, s1_root: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    s1_path = find_s1_raster(s1_root, city)

    nodata_mask, meta = build_s1_nodata_mask(
        s1_path=s1_path,
        all_zero_as_nodata=args.s1_all_zero_as_nodata,
        nan_as_nodata=args.nan_as_nodata,
    )

    nodata_pixels = int(nodata_mask.sum())

    base_row = {
        "city": city,
        "s1_path": str(s1_path),
        "width": meta["width"],
        "height": meta["height"],
        "band_count": meta["band_count"],
        "dtype": meta["dtype"],
        "crs": str(meta["crs"]),
        "nodata_value": meta["nodata_value"],
        "total_pixels": meta["total_pixels"],
        "s1_official_nodata_pixels": meta["official_nodata_pixels"],
        "s1_all_zero_allbands_pixels": meta["all_zero_allbands_pixels"],
        "s1_nonfinite_pixels": meta["nonfinite_pixels"],
        "s1_combined_nodata_pixels": meta["combined_nodata_pixels"],
        "s1_official_nodata_percent": meta["official_nodata_percent"],
        "s1_all_zero_allbands_percent": meta["all_zero_allbands_percent"],
        "s1_nonfinite_percent": meta["nonfinite_percent"],
        "s1_combined_nodata_percent": meta["combined_nodata_percent"],
        "min_area_pixels": args.min_area_pixels,
        "buffer_meters": args.buffer_meters,
        "simplify_tolerance_meters": args.simplify_tolerance_meters,
    }

    if nodata_pixels == 0:
        return {
            **base_row,
            "status": "no_s1_nodata",
            "missing_polygon_count_before_dissolve": 0,
            "missing_area_native": 0.0,
            "native_output_path": "",
            "wgs84_output_path": "",
            "geometry": None,
            "crs_native": meta["crs"],
        }

    transform = meta["transform"]
    pixel_area_native = abs(transform.a * transform.e)

    polygons = polygonize_nodata_mask(
        nodata_mask=nodata_mask,
        transform=transform,
        pixel_area_native=pixel_area_native,
        min_area_pixels=args.min_area_pixels,
    )

    if not polygons:
        return {
            **base_row,
            "status": "nodata_detected_but_no_polygon_after_filter",
            "missing_polygon_count_before_dissolve": 0,
            "missing_area_native": 0.0,
            "native_output_path": "",
            "wgs84_output_path": "",
            "geometry": None,
            "crs_native": meta["crs"],
        }

    dissolved = clean_geometry(unary_union(polygons))

    if args.buffer_meters != 0:
        dissolved = clean_geometry(dissolved.buffer(args.buffer_meters))

    if args.simplify_tolerance_meters > 0:
        dissolved = clean_geometry(
            dissolved.simplify(args.simplify_tolerance_meters, preserve_topology=True)
        )

    missing_area_native = float(dissolved.area)

    row_attrs = {
        **base_row,
        "status": "aoi_extracted",
        "missing_polygon_count_before_dissolve": len(polygons),
        "missing_area_native": missing_area_native,
    }

    native_path, wgs84_path, row_extra = write_city_outputs(
        gpd=gpd,
        city=city,
        geom_native=dissolved,
        crs_native=meta["crs"],
        row_attrs=row_attrs,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )

    return {
        **row_attrs,
        **row_extra,
        "geometry": dissolved,
        "crs_native": meta["crs"],
    }


def main() -> None:
    args = parse_args()
    gpd, _ = require_geopandas()

    instance_root = Path(args.instance_root)
    s1_root = instance_root / args.s1_subdir

    if not instance_root.exists():
        raise FileNotFoundError(f"Instance root does not exist: {instance_root}")

    if not s1_root.exists():
        raise FileNotFoundError(f"S1 root does not exist: {s1_root}")

    output_dir = Path(args.output_dir) if args.output_dir else instance_root / "qc" / "s1_missing_aoi"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "native").mkdir(exist_ok=True)
    (output_dir / "wgs84").mkdir(exist_ok=True)

    cities = discover_cities(s1_root, args.cities)

    print(f"[INFO] Instance root: {instance_root}")
    print(f"[INFO] S1 root: {s1_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Cities to process: {len(cities)}")
    print(f"[INFO] s1_all_zero_as_nodata: {args.s1_all_zero_as_nodata}")
    print(f"[INFO] nan_as_nodata: {args.nan_as_nodata}")
    print(f"[INFO] min_area_pixels: {args.min_area_pixels}")
    print(f"[INFO] buffer_meters: {args.buffer_meters}")
    print(f"[INFO] simplify_tolerance_meters: {args.simplify_tolerance_meters}")

    rows = []

    for i, city in enumerate(cities, start=1):
        print(f"\n[STEP {i}/{len(cities)}] {city}")

        try:
            row = process_city(
                gpd=gpd,
                city=city,
                s1_root=s1_root,
                output_dir=output_dir,
                args=args,
            )
            rows.append(row)

            print(
                "[OK] "
                f"status={row['status']} | "
                f"S1 nodata={row['s1_combined_nodata_percent']:.6f}% | "
                f"polygons={row.get('missing_polygon_count_before_dissolve', '')} | "
                f"wgs84={row.get('wgs84_output_path', '')}"
            )

        except Exception as exc:
            print(f"[ERROR] {city}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "city": city,
                    "status": "error",
                    "error": str(exc),
                    "geometry": None,
                    "crs_native": None,
                }
            )

    # Aggregate only works safely when all geometries share CRS. The city-level
    # WGS84 GeoJSON files are the most important outputs.
    try:
        write_aggregate_outputs(gpd, rows, output_dir, overwrite=args.overwrite)
    except Exception as exc:
        print(f"[WARN] Could not write aggregate AOI outputs: {exc}", file=sys.stderr)

    csv_path = output_dir / "s1_missing_aoi_summary.csv"
    json_path = output_dir / "s1_missing_aoi_summary.json"
    md_path = output_dir / "s1_missing_aoi_summary.md"

    write_csv(rows, csv_path, overwrite=args.overwrite)
    write_json(rows, json_path, overwrite=args.overwrite)
    write_markdown(rows, md_path, overwrite=args.overwrite)

    print("\n[DONE] Wrote:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print(f"  Per-city WGS84 AOIs: {output_dir / 'wgs84'}")
    print(f"  Per-city native AOIs: {output_dir / 'native'}")

    print("\n[SUMMARY]")
    counts = {}
    for row in rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()