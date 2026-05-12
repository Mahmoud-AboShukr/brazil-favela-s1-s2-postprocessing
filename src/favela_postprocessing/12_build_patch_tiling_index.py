#!/usr/bin/env python3
"""
Build patch tiling index for the Brazil favela segmentation dataset.

Purpose
-------
This script converts city-level aligned rasters into a patch/window index.

It does NOT read or export pixel arrays. It only defines patch windows:
    - city
    - split assignment
    - row/column window
    - patch bounds
    - patch centroid
    - source raster paths

This is the bridge between the geospatial products and ML training.

Inputs
------
Instance B products:
    S2 reflectance:
        <output_root>/dataset_instances/instance_B_standard_rs/s2/<city>/
            <city>_s2_12bands_reflectance_10m.tif

    SNAP S1:
        <output_root>/dataset_instances/instance_B_standard_rs/s1_snap/<city>/
            <city>_s1_snap_vv_vh_vvdiff_10m_aligned.tif

    Labels:
        <output_root>/labels_final/<city>/<city>_label_final.tif

Split file:
    <output_root>/metadata/split_train_covered_region_test.csv

Outputs
-------
CSV:
    <output_root>/metadata/patch_tiling_index_<strategy>_ps<patch>_st<stride>.csv

GeoPackage, if geopandas/shapely are available:
    <output_root>/metadata/patch_tiling_index_<strategy>_ps<patch>_st<stride>.gpkg

Summary:
    <output_root>/qc/patch_tiling_index_summary_<strategy>_ps<patch>_st<stride>.csv
    <output_root>/metadata/patch_tiling_index_<strategy>_ps<patch>_st<stride>.md

Example
-------
Recommended first tiling:

    python3 src/favela_postprocessing/12_build_patch_tiling_index.py --config configs/default.yaml

512 x 512 with 50 percent overlap:

    python3 src/favela_postprocessing/12_build_patch_tiling_index.py --config configs/default.yaml --patch-size 512 --overlap 0.5

Use leave-one-region-out split:

    python3 src/favela_postprocessing/12_build_patch_tiling_index.py --config configs/default.yaml --split-strategy leave_one_region_out
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import rasterio
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from rasterio.warp import transform_bounds, transform as transform_points
import yaml
from tqdm import tqdm


SCRIPT_NAME = "12_build_patch_tiling_index.py"
INSTANCE_NAME = "instance_B_standard_rs"

SPLIT_FILES = {
    "train_covered_region_test": "split_train_covered_region_test.csv",
    "balanced_5val_5test": "split_balanced_5val_5test.csv",
    "leave_one_region_out": "split_leave_one_region_out.csv",
    "all_cities_train": "split_all_cities_train.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build patch tiling index for aligned city-level rasters."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="train_covered_region_test",
        choices=sorted(SPLIT_FILES.keys()),
        help="Split strategy to attach to patches. Default: train_covered_region_test",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Optional explicit split CSV path. Overrides --split-strategy file lookup.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Patch size in pixels. Default: 512",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride in pixels. If omitted, computed from patch size and overlap.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap fraction in [0, 1). Used only when --stride is not provided. Default: 0.0",
    )
    parser.add_argument(
        "--edge-mode",
        type=str,
        default="cover",
        choices=["cover", "drop", "partial"],
        help=(
            "How to handle city borders. "
            "'cover' adds full-size edge patches shifted inward; "
            "'drop' keeps only regular full patches; "
            "'partial' allows smaller border patches. "
            "Default: cover."
        ),
    )
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="Process only one city. Can be repeated.",
    )
    parser.add_argument(
        "--no-gpkg",
        action="store_true",
        help="Do not write GeoPackage output.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "output_root" not in cfg:
        raise KeyError("Missing required key in config: output_root")

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


def bool_from_any(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def compute_stride(patch_size: int, stride: Optional[int], overlap: float) -> int:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    if stride is not None:
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        return int(stride)

    if overlap < 0 or overlap >= 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    computed = int(round(patch_size * (1.0 - overlap)))

    if computed <= 0:
        raise ValueError(
            f"Computed stride is invalid: patch_size={patch_size}, overlap={overlap}, stride={computed}"
        )

    return computed


def compute_starts(
    length: int,
    patch_size: int,
    stride: int,
    edge_mode: str,
) -> List[int]:
    """
    Compute row/column starts.

    edge_mode='cover':
        Full-size patches only. Add final shifted patch so the city edge is covered.

    edge_mode='drop':
        Full-size patches only. Drop border remainder.

    edge_mode='partial':
        Allow border patches smaller than patch_size.
    """
    if length <= 0:
        return []

    if edge_mode == "partial":
        return list(range(0, length, stride))

    if length < patch_size:
        return []

    starts = list(range(0, length - patch_size + 1, stride))

    if not starts:
        starts = [0]

    if edge_mode == "cover":
        last_start = length - patch_size
        if starts[-1] != last_start:
            starts.append(last_start)

    return sorted(set(starts))


def split_file_path(output_root: Path, split_strategy: str, explicit_split_file: Optional[Path]) -> Path:
    if explicit_split_file is not None:
        return explicit_split_file

    filename = SPLIT_FILES[split_strategy]
    return output_root / "metadata" / filename


def load_split_table(
    output_root: Path,
    split_strategy: str,
    explicit_split_file: Optional[Path],
    selected_cities: Optional[Sequence[str]],
) -> pd.DataFrame:
    path = split_file_path(output_root, split_strategy, explicit_split_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            f"Run 11_build_geographic_splits.py first."
        )

    df = pd.read_csv(path)

    required = [
        "city",
        "city_name",
        "state",
        "state_abbrev",
        "region",
        "region_order",
        "split_strategy",
        "fold_id",
        "heldout_region",
        "split",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Split table is missing required columns: {missing}")

    df = df.copy()
    df["city"] = df["city"].map(normalize_city_name)

    if selected_cities:
        selected = {normalize_city_name(city) for city in selected_cities}
        df = df[df["city"].isin(selected)].copy()

    if df.empty:
        raise RuntimeError("No split rows selected.")

    return df.reset_index(drop=True)


def derive_s2_path(output_root: Path, city: str) -> Path:
    return (
        output_root
        / "dataset_instances"
        / INSTANCE_NAME
        / "s2"
        / city
        / f"{city}_s2_12bands_reflectance_10m.tif"
    )


def derive_s1_path(output_root: Path, city: str) -> Path:
    return (
        output_root
        / "dataset_instances"
        / INSTANCE_NAME
        / "s1_snap"
        / city
        / f"{city}_s1_snap_vv_vh_vvdiff_10m_aligned.tif"
    )


def derive_label_path(output_root: Path, city: str) -> Path:
    return output_root / "labels_final" / city / f"{city}_label_final.tif"


def get_path_from_row(row: pd.Series, col: str, fallback: Path) -> Path:
    value = row.get(col, "")

    if isinstance(value, str) and value.strip():
        return Path(value)

    return fallback


def patch_id_from_parts(
    split_strategy: str,
    fold_id: str,
    city: str,
    row_start: int,
    col_start: int,
    patch_height: int,
    patch_width: int,
) -> str:
    safe_fold = str(fold_id).replace(" ", "_").replace("-", "_")
    return (
        f"{split_strategy}__{safe_fold}__{city}"
        f"__r{row_start:06d}_c{col_start:06d}"
        f"__h{patch_height:04d}_w{patch_width:04d}"
    )


def transform_bbox_to_lonlat(
    src_crs: Any,
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> Tuple[float, float, float, float]:
    if src_crs is None:
        return (math.nan, math.nan, math.nan, math.nan)

    try:
        return transform_bounds(
            src_crs,
            "EPSG:4326",
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )
    except Exception:
        return (math.nan, math.nan, math.nan, math.nan)


def transform_point_to_lonlat(src_crs: Any, x: float, y: float) -> Tuple[float, float]:
    if src_crs is None:
        return (math.nan, math.nan)

    try:
        lon_values, lat_values = transform_points(src_crs, "EPSG:4326", [x], [y])
        return float(lon_values[0]), float(lat_values[0])
    except Exception:
        return (math.nan, math.nan)


def build_patches_for_split_row(
    output_root: Path,
    row: pd.Series,
    patch_size: int,
    stride: int,
    edge_mode: str,
) -> List[Dict[str, Any]]:
    city = normalize_city_name(str(row["city"]))

    s2_path = get_path_from_row(
        row,
        "s2_reflectance_path",
        derive_s2_path(output_root, city),
    )

    s1_path = get_path_from_row(
        row,
        "s1_snap_path",
        derive_s1_path(output_root, city),
    )

    label_path = get_path_from_row(
        row,
        "label_path",
        derive_label_path(output_root, city),
    )

    if not s2_path.exists():
        raise FileNotFoundError(f"S2 reflectance file not found for {city}: {s2_path}")

    if not s1_path.exists():
        raise FileNotFoundError(f"SNAP S1 file not found for {city}: {s1_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found for {city}: {label_path}")

    patches: List[Dict[str, Any]] = []

    with rasterio.open(s2_path) as src:
        height = src.height
        width = src.width
        crs = src.crs
        full_transform = src.transform

        row_starts = compute_starts(
            length=height,
            patch_size=patch_size,
            stride=stride,
            edge_mode=edge_mode,
        )

        col_starts = compute_starts(
            length=width,
            patch_size=patch_size,
            stride=stride,
            edge_mode=edge_mode,
        )

        for row_start in row_starts:
            for col_start in col_starts:
                if edge_mode == "partial":
                    patch_height = min(patch_size, height - row_start)
                    patch_width = min(patch_size, width - col_start)
                else:
                    patch_height = patch_size
                    patch_width = patch_size

                row_end = row_start + patch_height
                col_end = col_start + patch_width

                window = Window(
                    col_off=col_start,
                    row_off=row_start,
                    width=patch_width,
                    height=patch_height,
                )

                left, bottom, right, top = window_bounds(window, full_transform)
                patch_transform = window_transform(window, full_transform)

                centroid_x = (left + right) / 2.0
                centroid_y = (bottom + top) / 2.0
                centroid_lon, centroid_lat = transform_point_to_lonlat(crs, centroid_x, centroid_y)

                lon_min, lat_min, lon_max, lat_max = transform_bbox_to_lonlat(
                    crs,
                    left,
                    bottom,
                    right,
                    top,
                )

                split_strategy = str(row["split_strategy"])
                fold_id = str(row["fold_id"])

                patch_id = patch_id_from_parts(
                    split_strategy=split_strategy,
                    fold_id=fold_id,
                    city=city,
                    row_start=row_start,
                    col_start=col_start,
                    patch_height=patch_height,
                    patch_width=patch_width,
                )

                is_edge_patch = bool(
                    row_start == 0
                    or col_start == 0
                    or row_end == height
                    or col_end == width
                )

                patches.append(
                    {
                        "patch_id": patch_id,
                        "city": city,
                        "city_name": row["city_name"],
                        "state": row["state"],
                        "state_abbrev": row["state_abbrev"],
                        "region": row["region"],
                        "region_order": row["region_order"],
                        "split_strategy": split_strategy,
                        "fold_id": fold_id,
                        "heldout_region": row.get("heldout_region", ""),
                        "split": row["split"],
                        "patch_size_requested": patch_size,
                        "stride": stride,
                        "edge_mode": edge_mode,
                        "source_s2_path": str(s2_path),
                        "source_s1_path": str(s1_path),
                        "source_label_path": str(label_path),
                        "city_raster_height": height,
                        "city_raster_width": width,
                        "patch_height": patch_height,
                        "patch_width": patch_width,
                        "row_start": row_start,
                        "row_end": row_end,
                        "col_start": col_start,
                        "col_end": col_end,
                        "is_full_size_patch": bool(patch_height == patch_size and patch_width == patch_size),
                        "is_edge_patch": is_edge_patch,
                        "crs": str(crs),
                        "city_transform": str(full_transform),
                        "patch_transform": str(patch_transform),
                        "bbox_minx": left,
                        "bbox_miny": bottom,
                        "bbox_maxx": right,
                        "bbox_maxy": top,
                        "bbox_minlon": lon_min,
                        "bbox_minlat": lat_min,
                        "bbox_maxlon": lon_max,
                        "bbox_maxlat": lat_max,
                        "centroid_x": centroid_x,
                        "centroid_y": centroid_y,
                        "centroid_lon": centroid_lon,
                        "centroid_lat": centroid_lat,
                    }
                )

    return patches


def safe_suffix(split_strategy: str, patch_size: int, stride: int, edge_mode: str) -> str:
    return f"{split_strategy}_ps{patch_size}_st{stride}_{edge_mode}"


def summarize_patch_index(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "split_strategy",
        "fold_id",
        "heldout_region",
        "split",
        "region",
    ]

    summary = (
        df.groupby(group_cols)
        .agg(
            patch_count=("patch_id", "count"),
            city_count=("city", "nunique"),
            cities=("city", lambda x: ",".join(sorted(set(x.astype(str))))),
            full_size_patch_count=("is_full_size_patch", "sum"),
            edge_patch_count=("is_edge_patch", "sum"),
        )
        .reset_index()
        .sort_values(["split_strategy", "fold_id", "split", "region"])
    )

    return summary


def markdown_escape(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return "nan"
    except Exception:
        pass

    text = str(value)
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


def df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    columns = list(df.columns)

    header = "| " + " | ".join(markdown_escape(col) for col in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    rows = []

    for _, row in df.iterrows():
        rows.append(
            "| "
            + " | ".join(markdown_escape(row[col]) for col in columns)
            + " |"
        )

    return "\n".join([header, separator] + rows)


def write_markdown_summary(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    md_path: Path,
    patch_size: int,
    stride: int,
    overlap: float,
    edge_mode: str,
    split_strategy: str,
) -> None:
    ensure_dir(md_path.parent)

    lines: List[str] = []

    lines.append("# Patch Tiling Index Summary")
    lines.append("")
    lines.append(f"Generated by `{SCRIPT_NAME}`.")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "This file defines spatial patch windows over the aligned city-level rasters. "
        "It does not export pixel arrays. It only records where each patch is located "
        "and which split assignment it inherits from its city."
    )
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    param_df = pd.DataFrame(
        [
            {"parameter": "split_strategy", "value": split_strategy},
            {"parameter": "patch_size", "value": patch_size},
            {"parameter": "stride", "value": stride},
            {"parameter": "overlap", "value": overlap},
            {"parameter": "edge_mode", "value": edge_mode},
            {"parameter": "total_patches", "value": len(df)},
            {"parameter": "cities", "value": df["city"].nunique()},
            {"parameter": "folds", "value": df["fold_id"].nunique()},
        ]
    )
    lines.append(df_to_markdown_table(param_df))
    lines.append("")

    lines.append("## Patch counts by split and region")
    lines.append("")
    display_summary = summary_df[
        [
            "split_strategy",
            "fold_id",
            "heldout_region",
            "split",
            "region",
            "city_count",
            "patch_count",
        ]
    ].copy()
    lines.append(df_to_markdown_table(display_summary))
    lines.append("")

    lines.append("## City-level patch counts")
    lines.append("")
    city_counts = (
        df.groupby(["fold_id", "split", "region", "city"])
        .size()
        .reset_index(name="patch_count")
        .sort_values(["fold_id", "split", "region", "city"])
    )
    lines.append(df_to_markdown_table(city_counts))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Each row in the CSV is one patch window.")
    lines.append("- `row_start`, `row_end`, `col_start`, and `col_end` define the pixel window inside the source city rasters.")
    lines.append("- `bbox_*` columns define the native CRS bounding box.")
    lines.append("- `bbox_*lon/lat` and `centroid_lon/lat` provide WGS84 coordinates for mapping and inspection.")
    lines.append("- Pixel-level statistics such as label percentage, cloud percentage, and nodata percentage are not computed here; they belong to the next script.")
    lines.append("")

    lines.append("## Recommended next step")
    lines.append("")
    lines.append(
        "Run the patch metadata computation script to calculate label-positive percentage, "
        "nodata percentage, and optional cloud-quality indicators for each patch."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_gpkg(df: pd.DataFrame, gpkg_path: Path) -> bool:
    """
    Write patch polygons in EPSG:4326.

    Returns True if successful, False if geopandas/shapely are unavailable.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except Exception as exc:
        print(f"[WARN] Could not import geopandas/shapely. Skipping GeoPackage. Reason: {exc}")
        return False

    ensure_dir(gpkg_path.parent)

    geometry = [
        box(row.bbox_minlon, row.bbox_minlat, row.bbox_maxlon, row.bbox_maxlat)
        for row in df.itertuples(index=False)
    ]

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=geometry,
        crs="EPSG:4326",
    )

    gdf.to_file(gpkg_path, driver="GPKG")
    return True


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))
    metadata_root = output_root / "metadata"
    qc_root = output_root / "qc"

    ensure_dir(metadata_root)
    ensure_dir(qc_root)

    stride = compute_stride(
        patch_size=args.patch_size,
        stride=args.stride,
        overlap=args.overlap,
    )

    suffix = safe_suffix(
        split_strategy=args.split_strategy,
        patch_size=args.patch_size,
        stride=stride,
        edge_mode=args.edge_mode,
    )

    csv_path = metadata_root / f"patch_tiling_index_{suffix}.csv"
    gpkg_path = metadata_root / f"patch_tiling_index_{suffix}.gpkg"
    md_path = metadata_root / f"patch_tiling_index_{suffix}.md"
    summary_path = qc_root / f"patch_tiling_index_summary_{suffix}.csv"

    print("[INFO] Build patch tiling index")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Split strategy: {args.split_strategy}")
    print(f"[INFO] Patch size: {args.patch_size}")
    print(f"[INFO] Stride: {stride}")
    print(f"[INFO] Overlap: {args.overlap}")
    print(f"[INFO] Edge mode: {args.edge_mode}")
    print(f"[INFO] CSV output: {csv_path}")
    print(f"[INFO] GPKG output: {gpkg_path}")
    print(f"[INFO] Summary output: {summary_path}")

    split_df = load_split_table(
        output_root=output_root,
        split_strategy=args.split_strategy,
        explicit_split_file=args.split_file,
        selected_cities=args.city,
    )

    print(f"[INFO] Split rows selected: {len(split_df)}")
    print(f"[INFO] Cities selected: {split_df['city'].nunique()}")
    print(f"[INFO] Folds selected: {split_df['fold_id'].nunique()}")

    all_patches: List[Dict[str, Any]] = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Building patch windows"):
        city = normalize_city_name(str(row["city"]))
        fold_id = str(row["fold_id"])
        split = str(row["split"])

        patches = build_patches_for_split_row(
            output_root=output_root,
            row=row,
            patch_size=args.patch_size,
            stride=stride,
            edge_mode=args.edge_mode,
        )

        print(f"[INFO] {city} | fold={fold_id} | split={split}: {len(patches)} patches")

        all_patches.extend(patches)

    if not all_patches:
        raise RuntimeError("No patches were generated.")

    df = pd.DataFrame(all_patches)

    df = df.sort_values(
        [
            "split_strategy",
            "fold_id",
            "split",
            "region_order",
            "city",
            "row_start",
            "col_start",
        ]
    ).reset_index(drop=True)

    df.to_csv(csv_path, index=False)

    summary_df = summarize_patch_index(df)
    summary_df.to_csv(summary_path, index=False)

    write_markdown_summary(
        df=df,
        summary_df=summary_df,
        md_path=md_path,
        patch_size=args.patch_size,
        stride=stride,
        overlap=args.overlap,
        edge_mode=args.edge_mode,
        split_strategy=args.split_strategy,
    )

    gpkg_written = False

    if not args.no_gpkg:
        gpkg_written = write_gpkg(df, gpkg_path)

    print(f"[INFO] Wrote CSV: {csv_path}")
    print(f"[INFO] Wrote summary CSV: {summary_path}")
    print(f"[INFO] Wrote Markdown: {md_path}")

    if gpkg_written:
        print(f"[INFO] Wrote GeoPackage: {gpkg_path}")
    else:
        print("[INFO] GeoPackage not written.")

    print("[INFO] Patch count summary by split:")
    print(
        df.groupby(["split", "region"])
        .size()
        .rename("patch_count")
        .reset_index()
        .sort_values(["split", "region"])
        .to_string(index=False)
    )

    print(f"[INFO] Total patches: {len(df)}")
    print("[INFO] Patch tiling index built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())