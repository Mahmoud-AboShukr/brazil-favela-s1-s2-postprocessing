#!/usr/bin/env python3
"""
Build QC summary for Instance B: standard remote-sensing dataset.

Instance B contains:

    S2 reflectance:
        <output_root>/dataset_instances/instance_B_standard_rs/s2/<city>/
            <city>_s2_12bands_reflectance_10m.tif

    SNAP terrain-corrected S1:
        <output_root>/dataset_instances/instance_B_standard_rs/s1_snap/<city>/
            <city>_s1_snap_vv_vh_vvdiff_10m_aligned.tif

    Final binary labels:
        <output_root>/labels_final/<city>/<city>_label_final.tif

This script checks:
    - official city list
    - S2 reflectance existence
    - SNAP S1 existence
    - final label existence
    - S2/S1/label CRS equality
    - S2/S1/label transform equality
    - S2/S1/label shape equality
    - expected band counts
    - complete Instance B city count

Outputs:
    <output_root>/qc/instance_B_standard_rs_summary.csv
    <output_root>/qc/instance_B_standard_rs_summary.md

Example:
    python src/favela_postprocessing/09_build_instance_b_standard_rs_summary.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import rasterio
import yaml
from tqdm import tqdm


SCRIPT_NAME = "09_build_instance_b_standard_rs_summary.py"
INSTANCE_NAME = "instance_B_standard_rs"

EXPECTED_S2_BANDS = 12
EXPECTED_S1_BANDS = 3
EXPECTED_LABEL_BANDS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Instance B standard remote-sensing dataset QC summary."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="Check only one city. Can be repeated.",
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


def discover_official_cities(output_root: Path, selected_cities: Optional[Sequence[str]]) -> List[str]:
    if selected_cities:
        return sorted(set(normalize_city_name(city) for city in selected_cities))

    s2_final_root = output_root / "s2_final"

    if not s2_final_root.exists():
        raise FileNotFoundError(
            f"Could not discover official city list because s2_final does not exist: {s2_final_root}"
        )

    cities = sorted(
        path.name
        for path in s2_final_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )

    if not cities:
        raise RuntimeError(f"No city folders found under: {s2_final_root}")

    return cities


def s2_reflectance_path(output_root: Path, city: str) -> Path:
    return (
        output_root
        / "dataset_instances"
        / INSTANCE_NAME
        / "s2"
        / city
        / f"{city}_s2_12bands_reflectance_10m.tif"
    )


def s1_snap_path(output_root: Path, city: str) -> Path:
    return (
        output_root
        / "dataset_instances"
        / INSTANCE_NAME
        / "s1_snap"
        / city
        / f"{city}_s1_snap_vv_vh_vvdiff_10m_aligned.tif"
    )


def label_path(output_root: Path, city: str) -> Path:
    return output_root / "labels_final" / city / f"{city}_label_final.tif"


def transform_equal(a: Any, b: Any, tolerance: float = 1e-9) -> bool:
    try:
        return all(abs(x - y) <= tolerance for x, y in zip(a, b))
    except Exception:
        return False


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return math.nan
    return path.stat().st_size / (1024.0 * 1024.0)


def read_raster_meta(path: Path, prefix: str) -> Dict[str, Any]:
    if not path.exists():
        return {
            f"{prefix}_exists": False,
            f"{prefix}_path": str(path),
            f"{prefix}_error": "missing_file",
        }

    try:
        with rasterio.open(path) as src:
            res_x = float(abs(src.transform.a))
            res_y = float(abs(src.transform.e))

            return {
                f"{prefix}_exists": True,
                f"{prefix}_path": str(path),
                f"{prefix}_size_mb": file_size_mb(path),
                f"{prefix}_band_count": src.count,
                f"{prefix}_dtypes": ",".join(src.dtypes),
                f"{prefix}_crs": str(src.crs),
                f"{prefix}_width": src.width,
                f"{prefix}_height": src.height,
                f"{prefix}_shape": f"{src.height}x{src.width}",
                f"{prefix}_transform": str(src.transform),
                f"{prefix}_resolution_x": res_x,
                f"{prefix}_resolution_y": res_y,
                f"{prefix}_nodata": src.nodata,
                f"{prefix}_descriptions": ",".join(desc or "" for desc in src.descriptions),
                f"{prefix}_error": "",
                f"_{prefix}_crs_obj": src.crs,
                f"_{prefix}_transform_obj": src.transform,
            }

    except Exception as exc:
        return {
            f"{prefix}_exists": True,
            f"{prefix}_path": str(path),
            f"{prefix}_size_mb": file_size_mb(path),
            f"{prefix}_error": repr(exc),
        }


def compare_pair(
    left_meta: Dict[str, Any],
    right_meta: Dict[str, Any],
    left_prefix: str,
    right_prefix: str,
    pair_name: str,
) -> Dict[str, Any]:
    left_exists = bool(left_meta.get(f"{left_prefix}_exists", False))
    right_exists = bool(right_meta.get(f"{right_prefix}_exists", False))

    if not left_exists or not right_exists:
        return {
            f"{pair_name}_alignment_ok": False,
            f"{pair_name}_same_crs": False,
            f"{pair_name}_same_transform": False,
            f"{pair_name}_same_shape": False,
            f"{pair_name}_same_resolution": False,
        }

    same_crs = left_meta.get(f"_{left_prefix}_crs_obj") == right_meta.get(f"_{right_prefix}_crs_obj")
    same_transform = transform_equal(
        left_meta.get(f"_{left_prefix}_transform_obj"),
        right_meta.get(f"_{right_prefix}_transform_obj"),
    )
    same_width = left_meta.get(f"{left_prefix}_width") == right_meta.get(f"{right_prefix}_width")
    same_height = left_meta.get(f"{left_prefix}_height") == right_meta.get(f"{right_prefix}_height")
    same_shape = bool(same_width and same_height)

    same_res_x = left_meta.get(f"{left_prefix}_resolution_x") == right_meta.get(f"{right_prefix}_resolution_x")
    same_res_y = left_meta.get(f"{left_prefix}_resolution_y") == right_meta.get(f"{right_prefix}_resolution_y")
    same_resolution = bool(same_res_x and same_res_y)

    alignment_ok = bool(same_crs and same_transform and same_shape and same_resolution)

    return {
        f"{pair_name}_alignment_ok": alignment_ok,
        f"{pair_name}_same_crs": bool(same_crs),
        f"{pair_name}_same_transform": bool(same_transform),
        f"{pair_name}_same_shape": bool(same_shape),
        f"{pair_name}_same_resolution": bool(same_resolution),
    }


def clean_internal_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
    }


def check_city(output_root: Path, city: str) -> Dict[str, Any]:
    s2_path = s2_reflectance_path(output_root, city)
    s1_path = s1_snap_path(output_root, city)
    lab_path = label_path(output_root, city)

    s2_meta = read_raster_meta(s2_path, "s2")
    s1_meta = read_raster_meta(s1_path, "s1_snap")
    label_meta = read_raster_meta(lab_path, "label")

    row: Dict[str, Any] = {
        "city": city,
        "s2_expected_path": str(s2_path),
        "s1_snap_expected_path": str(s1_path),
        "label_expected_path": str(lab_path),
    }

    row.update(s2_meta)
    row.update(s1_meta)
    row.update(label_meta)

    row.update(compare_pair(s2_meta, s1_meta, "s2", "s1_snap", "s2_s1_snap"))
    row.update(compare_pair(s2_meta, label_meta, "s2", "label", "s2_label"))

    s2_band_ok = row.get("s2_band_count") == EXPECTED_S2_BANDS
    s1_band_ok = row.get("s1_snap_band_count") == EXPECTED_S1_BANDS
    label_band_ok = row.get("label_band_count") == EXPECTED_LABEL_BANDS

    complete = bool(
        row.get("s2_exists", False)
        and row.get("s1_snap_exists", False)
        and row.get("label_exists", False)
        and s2_band_ok
        and s1_band_ok
        and label_band_ok
        and row.get("s2_s1_snap_alignment_ok", False)
        and row.get("s2_label_alignment_ok", False)
    )

    row.update(
        {
            "s2_band_count_ok": bool(s2_band_ok),
            "s1_snap_band_count_ok": bool(s1_band_ok),
            "label_band_count_ok": bool(label_band_ok),
            "complete_instance_B_standard_rs": complete,
        }
    )

    if complete:
        status = "COMPLETE"
    elif not row.get("s2_exists", False):
        status = "MISSING_S2"
    elif not row.get("s1_snap_exists", False):
        status = "MISSING_S1_SNAP"
    elif not row.get("label_exists", False):
        status = "MISSING_LABEL"
    elif not row.get("s2_s1_snap_alignment_ok", False):
        status = "S2_S1_ALIGNMENT_FAIL"
    elif not row.get("s2_label_alignment_ok", False):
        status = "S2_LABEL_ALIGNMENT_FAIL"
    else:
        status = "INCOMPLETE_OTHER"

    row["status"] = status

    return clean_internal_meta(row)


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


def series_to_markdown_table(series: pd.Series, index_name: str, value_name: str) -> str:
    if series.empty:
        return "_No rows._"

    table = series.rename_axis(index_name).reset_index(name=value_name)
    return df_to_markdown_table(table)


def write_markdown_summary(df: pd.DataFrame, md_path: Path) -> None:
    ensure_dir(md_path.parent)

    lines: List[str] = []

    lines.append("# Instance B Standard Remote-Sensing Dataset Summary")
    lines.append("")
    lines.append(f"Generated by `{SCRIPT_NAME}`.")
    lines.append("")
    lines.append("## Definition")
    lines.append("")
    lines.append("Instance B is defined as:")
    lines.append("")
    lines.append("- 12-band Sentinel-2 reflectance products.")
    lines.append("- SNAP terrain-corrected Sentinel-1 products with `VV_dB`, `VH_dB`, and `VV_minus_VH_dB`.")
    lines.append("- Final refined binary favela label masks.")
    lines.append("")
    lines.append("## Status counts")
    lines.append("")
    lines.append(series_to_markdown_table(df["status"].value_counts(dropna=False), "status", "city_count"))
    lines.append("")

    total = len(df)
    s2_count = int(df["s2_exists"].sum()) if "s2_exists" in df.columns else 0
    s1_count = int(df["s1_snap_exists"].sum()) if "s1_snap_exists" in df.columns else 0
    label_count = int(df["label_exists"].sum()) if "label_exists" in df.columns else 0
    s2_s1_ok = int(df["s2_s1_snap_alignment_ok"].sum()) if "s2_s1_snap_alignment_ok" in df.columns else 0
    s2_label_ok = int(df["s2_label_alignment_ok"].sum()) if "s2_label_alignment_ok" in df.columns else 0
    complete_count = int(df["complete_instance_B_standard_rs"].sum()) if "complete_instance_B_standard_rs" in df.columns else 0

    lines.append("## Main counts")
    lines.append("")
    counts_df = pd.DataFrame(
        [
            {"check": "S2 reflectance products", "count": s2_count, "total": total},
            {"check": "SNAP S1 products", "count": s1_count, "total": total},
            {"check": "Final label masks", "count": label_count, "total": total},
            {"check": "S2-SNAP S1 alignment OK", "count": s2_s1_ok, "total": total},
            {"check": "S2-label alignment OK", "count": s2_label_ok, "total": total},
            {"check": "Complete Instance B cities", "count": complete_count, "total": total},
        ]
    )
    lines.append(df_to_markdown_table(counts_df))
    lines.append("")

    lines.append("## City-level summary")
    lines.append("")

    city_cols = [
        "city",
        "status",
        "s2_exists",
        "s1_snap_exists",
        "label_exists",
        "s2_band_count",
        "s1_snap_band_count",
        "label_band_count",
        "s2_s1_snap_alignment_ok",
        "s2_label_alignment_ok",
        "complete_instance_B_standard_rs",
    ]

    existing_city_cols = [col for col in city_cols if col in df.columns]
    city_df = df[existing_city_cols].copy()

    lines.append(df_to_markdown_table(city_df))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `COMPLETE` means the city has S2 reflectance, SNAP S1, and final labels, all aligned to the same grid.")
    lines.append("- `s2_s1_snap_alignment_ok=True` means CRS, transform, resolution, width, and height match between S2 and SNAP S1.")
    lines.append("- `s2_label_alignment_ok=True` means CRS, transform, resolution, width, and height match between S2 and the final binary label.")
    lines.append("")
    lines.append("## Recommended next step")
    lines.append("")
    lines.append(
        "If all 26 cities are complete, Instance B can be used as the standard remote-sensing source "
        "for the next stage: city-region metadata, split definitions, tiling index generation, patch metadata, "
        "normalization statistics, H5 export, and CROMA probing experiments."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))
    cities = discover_official_cities(output_root, args.city)

    csv_path = output_root / "qc" / "instance_B_standard_rs_summary.csv"
    md_path = output_root / "qc" / "instance_B_standard_rs_summary.md"

    print("[INFO] Instance B standard remote-sensing summary")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Cities selected: {len(cities)}")
    print(f"[INFO] CSV output: {csv_path}")
    print(f"[INFO] Markdown output: {md_path}")

    rows: List[Dict[str, Any]] = []

    for city in tqdm(cities, desc="Checking Instance B cities"):
        row = check_city(output_root, city)
        rows.append(row)
        print(f"[INFO] {city}: {row['status']}")

    df = pd.DataFrame(rows)

    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)

    write_markdown_summary(df, md_path)

    print(f"[INFO] Wrote CSV: {csv_path}")
    print(f"[INFO] Wrote Markdown summary: {md_path}")

    if "status" in df.columns:
        print("[INFO] Status counts:")
        print(df["status"].value_counts(dropna=False).to_string())

    total = len(df)
    complete = int(df["complete_instance_B_standard_rs"].sum())

    print(f"[INFO] Complete Instance B cities: {complete} / {total}")

    if complete != total:
        print("[ERROR] Instance B is incomplete. Check the CSV/Markdown summary.")
        return 1

    print("[INFO] Instance B is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())