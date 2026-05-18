#!/usr/bin/env python3
"""
Preprocess fallback Sentinel-1 SAFE products with SNAP and align them to S2-filled grid.

Inputs:
    instance_C_s2_nodata_repaired/
        fallback_s1/raw/<city>/<selected_item_id>/*.SAFE/manifest.safe
        s2_filled/<city>/<city>_s2_12bands_reflectance_10m.tif
        s1_snap/<city>/<city>_s1_snap_vv_vh_vvdiff_10m_aligned.tif

Outputs:
    instance_C_s2_nodata_repaired/
        fallback_s1/processed/<city>/<city>_s1_fallback_vv_vh_vvdiff_10m_aligned.tif

QA:
    qc/s1_fallback_preprocess/
        graphs/<city>_fallback_s1_snap_graph.xml
        logs/<city>_gpt_stdout.txt
        logs/<city>_gpt_stderr.txt
        masks/<city>_fallback_valid_mask.tif
        masks/<city>_original_s1_missing_mask.tif
        masks/<city>_original_missing_fillable_by_fallback_mask.tif
        s1_fallback_preprocess_summary.csv/json/md

This script:
    1. Locates downloaded fallback SAFE products.
    2. Runs SNAP GPT preprocessing.
    3. Aligns fallback VV/VH to the S2-filled target grid.
    4. Writes a 3-band fallback S1 raster: VV_dB, VH_dB, VV_minus_VH_dB.
    5. Computes how much original S1 nodata is fillable by the fallback product.

It does not merge original and fallback S1.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


NODATA_FLOAT = -9999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess fallback S1 SAFE products with SNAP and align to S2 grid."
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
        "--selection-csv",
        type=str,
        default=None,
        help=(
            "Selected fallback products CSV. If omitted, uses "
            "<instance-root>/qc/s1_fallback_selection/selected_s1_fallback_products.csv"
        ),
    )

    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help=(
            "Raw fallback SAFE root. If omitted, uses "
            "<instance-root>/fallback_s1/raw"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Processed fallback output root. If omitted, uses "
            "<instance-root>/fallback_s1/processed"
        ),
    )

    parser.add_argument(
        "--tmp-root",
        type=str,
        default=None,
        help=(
            "Temporary SNAP output root. If omitted, uses "
            "<instance-root>/fallback_s1/snap_tmp"
        ),
    )

    parser.add_argument(
        "--qc-dir",
        type=str,
        default=None,
        help=(
            "QC directory. If omitted, uses "
            "<instance-root>/qc/s1_fallback_preprocess"
        ),
    )

    parser.add_argument(
        "--s2-subdir",
        type=str,
        default="s2_filled",
        help="S2-filled subdirectory inside instance root. Default: s2_filled",
    )

    parser.add_argument(
        "--original-s1-subdir",
        type=str,
        default="s1_snap",
        help="Original cropped S1 subdirectory inside instance root. Default: s1_snap",
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional city list. If omitted, all rows in selection table are processed.",
    )

    parser.add_argument(
        "--gpt-path",
        type=str,
        required=True,
        help='Path to SNAP GPT executable, e.g. "C:/Program Files/esa-snap/bin/gpt.exe"',
    )

    parser.add_argument(
        "--snap-cache",
        type=str,
        default="4G",
        help="SNAP GPT cache size, e.g. 4G or 8G. Default: 4G",
    )

    parser.add_argument(
        "--progress-interval-seconds",
        type=int,
        default=10,
        help="Refresh SNAP GPT progress bar every N seconds. Default: 10.",
    )

    parser.add_argument(
        "--dem-name",
        type=str,
        default="Copernicus 30m Global DEM",
        help="DEM name for SNAP Terrain-Correction. Default: Copernicus 30m Global DEM",
    )

    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=10.0,
        help="SNAP terrain correction pixel spacing in metres. Default: 10.0",
    )

    parser.add_argument(
        "--snap-map-projection",
        choices=["target", "none"],
        default="target",
        help=(
            "If target, pass S2 CRS to SNAP Terrain-Correction. "
            "If none, let SNAP choose and rely on rasterio alignment afterwards."
        ),
    )

    parser.add_argument(
        "--fallback-all-zero-as-nodata",
        action="store_true",
        default=True,
        help=(
            "Treat pixels where fallback VV and VH are both exactly zero as invalid "
            "after warping. Default: True."
        ),
    )

    parser.add_argument(
        "--no-fallback-all-zero-as-nodata",
        dest="fallback_all_zero_as_nodata",
        action="store_false",
        help="Do not treat all-zero fallback pixels as invalid.",
    )

    parser.add_argument(
        "--original-s1-all-zero-as-nodata",
        action="store_true",
        default=False,
        help=(
            "Treat original S1 all-zero-all-band pixels as nodata when computing "
            "original missing mask. Default: False."
        ),
    )

    parser.add_argument(
        "--run-snap",
        action="store_true",
        default=True,
        help="Run SNAP GPT. Default: True.",
    )

    parser.add_argument(
        "--no-run-snap",
        dest="run_snap",
        action="store_false",
        help="Skip SNAP GPT and reuse existing temporary SNAP GeoTIFF.",
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary SNAP outputs after successful alignment.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser.parse_args()


def percent(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def safe_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def write_json(obj: Any, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(safe_jsonable(obj), f, indent=2, ensure_ascii=False)


def write_csv(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["no_rows"])
        return

    fields: list[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "city",
        "status",
        "selected_item_id",
        "fallback_valid_percent",
        "original_s1_missing_percent",
        "original_missing_fillable_percent_of_missing",
        "expected_remaining_original_s1_nodata_percent_after_merge",
        "output_path",
        "message",
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Fallback S1 SNAP preprocessing summary\n\n")
        f.write(
            "This report summarizes SNAP preprocessing of downloaded fallback S1 SAFE "
            "products and alignment to the S2-filled grid. It does not merge S1 rasters.\n\n"
        )

        f.write("## Status counts\n\n")
        counts: dict[str, int] = {}
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


def safe_unlink(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")


def load_selection_table(selection_csv: Path, cities: list[str] | None) -> pd.DataFrame:
    if not selection_csv.exists():
        raise FileNotFoundError(f"Selection CSV does not exist: {selection_csv}")

    df = pd.read_csv(selection_csv)

    required = {"city", "selected_item_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Selection CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["city"] = df["city"].astype(str)
    df["selected_item_id"] = df["selected_item_id"].astype(str)

    if cities:
        df = df[df["city"].isin(cities)].copy()

    df = df[df["selected_item_id"].str.len() > 0].copy()

    if df.empty:
        raise ValueError("No selected fallback rows to process.")

    return df


def find_s2_target(instance_root: Path, s2_subdir: str, city: str) -> Path:
    city_dir = instance_root / s2_subdir / city
    candidates = sorted(city_dir.glob(f"{city}_s2_12bands_reflectance_10m.tif"))
    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s2*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No S2 target raster found for {city}: {city_dir}")
    return candidates[0]


def find_original_s1(instance_root: Path, s1_subdir: str, city: str) -> Path:
    city_dir = instance_root / s1_subdir / city
    candidates = sorted(city_dir.glob(f"{city}_s1_snap_vv_vh_vvdiff_10m_aligned.tif"))
    if not candidates:
        candidates = sorted(city_dir.glob(f"{city}_s1*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No original S1 raster found for {city}: {city_dir}")
    return candidates[0]


def find_safe_manifest(raw_root: Path, city: str, selected_item_id: str) -> tuple[Path, Path]:
    search_root = raw_root / city / selected_item_id

    if not search_root.exists():
        raise FileNotFoundError(f"Fallback raw folder does not exist: {search_root}")

    manifests = sorted(search_root.rglob("manifest.safe"))

    if not manifests:
        raise FileNotFoundError(f"No manifest.safe found under {search_root}")

    preferred = []
    for manifest in manifests:
        safe_dir = manifest.parent
        if selected_item_id in safe_dir.name:
            preferred.append(manifest)

    manifest = preferred[0] if preferred else manifests[0]
    return manifest.parent, manifest


def crs_to_snap_projection(crs) -> str:
    epsg = crs.to_epsg()
    if epsg:
        return f"EPSG:{epsg}"
    return crs.to_wkt()


def build_snap_graph_xml(
    manifest_path: Path,
    output_tif_path: Path,
    target_crs,
    args: argparse.Namespace,
) -> str:
    manifest_xml = html.escape(str(manifest_path.as_posix()))
    output_xml = html.escape(str(output_tif_path.as_posix()))
    dem_name_xml = html.escape(args.dem_name)

    map_projection_xml = ""
    if args.snap_map_projection == "target":
        projection = html.escape(crs_to_snap_projection(target_crs))
        map_projection_xml = f"<mapProjection>{projection}</mapProjection>"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<graph id="fallback_s1_snap_preprocessing">
  <version>1.0</version>

  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{manifest_xml}</file>
    </parameters>
  </node>

  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>

  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <removeThermalNoise>true</removeThermalNoise>
      <reIntroduceThermalNoise>false</reIntroduceThermalNoise>
    </parameters>
  </node>

  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="ThermalNoiseRemoval"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
      <outputImageScaleInDb>false</outputImageScaleInDb>
    </parameters>
  </node>

  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <sourceBands>Sigma0_VV,Sigma0_VH</sourceBands>
      <demName>{dem_name_xml}</demName>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>{args.pixel_spacing}</pixelSpacingInMeter>
      {map_projection_xml}
      <nodataValueAtSea>false</nodataValueAtSea>
      <saveDEM>false</saveDEM>
      <saveLatLon>false</saveLatLon>
      <saveIncidenceAngleFromEllipsoid>false</saveIncidenceAngleFromEllipsoid>
      <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
      <saveProjectedLocalIncidenceAngle>false</saveProjectedLocalIncidenceAngle>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
      <applyRadiometricNormalization>false</applyRadiometricNormalization>
    </parameters>
  </node>

  <node id="LinearToFromdB">
    <operator>LinearToFromdB</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <sourceBands>Sigma0_VV,Sigma0_VH</sourceBands>
    </parameters>
  </node>

  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="LinearToFromdB"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{output_xml}</file>
      <formatName>GeoTIFF-BigTIFF</formatName>
    </parameters>
  </node>

</graph>
"""


def read_recent_log_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""

    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_chars:
                f.seek(-max_chars, 2)
            data = f.read()
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def parse_snap_percent_from_text(text: str) -> float | None:
    if not text:
        return None

    matches = re.findall(r"(?<!\d)(100(?:\.0+)?|[0-9]{1,2}(?:\.[0-9]+)?)\s*%", text)

    values = []
    for match in matches:
        try:
            value = float(match)
            if 0.0 <= value <= 100.0:
                values.append(value)
        except ValueError:
            continue

    if not values:
        return None

    return max(values)


def estimate_snap_stage_percent(text: str) -> tuple[float, str]:
    lower = text.lower()

    detected_percent = 3.0
    detected_stage = "starting"

    checks = [
        ("read", 8.0, "read"),
        ("apply-orbit-file", 18.0, "apply orbit file"),
        ("apply orbit", 18.0, "apply orbit file"),
        ("thermalnoiseremoval", 28.0, "thermal noise removal"),
        ("thermal noise", 28.0, "thermal noise removal"),
        ("calibration", 42.0, "calibration"),
        ("terrain-correction", 70.0, "terrain correction"),
        ("terrain correction", 70.0, "terrain correction"),
        ("lineartofromdb", 84.0, "linear to db"),
        ("linear", 84.0, "linear to db"),
        ("write", 92.0, "write geotiff"),
        ("geotiff", 92.0, "write geotiff"),
    ]

    for keyword, pct, stage in checks:
        if keyword in lower and pct >= detected_percent:
            detected_percent = pct
            detected_stage = stage

    return detected_percent, detected_stage


def make_progress_bar(percent_value: float, width: int = 32) -> str:
    pct = max(0.0, min(100.0, float(percent_value)))
    filled = int(round((pct / 100.0) * width))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {pct:6.2f}%"


def print_progress_line(
    percent_value: float,
    elapsed_minutes: float,
    stage: str,
    snap_output_size_mb: float,
    source: str,
) -> None:
    bar = make_progress_bar(percent_value)
    msg = (
        f"\r{bar} | elapsed={elapsed_minutes:6.1f} min | "
        f"stage={stage[:28]:28s} | "
        f"output={snap_output_size_mb:8.2f} MB | "
        f"{source[:18]:18s}"
    )
    print(msg, end="", flush=True)


def run_snap_gpt(
    gpt_path: Path,
    graph_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    snap_cache: str,
    progress_interval_seconds: int = 10,
    monitored_output_path: Path | None = None,
) -> int:
    cmd = [
        str(gpt_path),
        str(graph_path),
        "-c",
        snap_cache,
    ]

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    best_percent = 0.0
    last_stage = "starting"

    with stdout_path.open("w", encoding="utf-8", errors="ignore") as out, stderr_path.open(
        "w", encoding="utf-8", errors="ignore"
    ) as err:
        process = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=err,
        )

        print(f"[INFO] SNAP GPT started. PID={process.pid}")
        print(f"[INFO] stdout log: {stdout_path}")
        print(f"[INFO] stderr log: {stderr_path}")

        while True:
            returncode = process.poll()

            elapsed = time.time() - start
            elapsed_minutes = elapsed / 60.0

            stdout_text = read_recent_log_text(stdout_path)
            stderr_text = read_recent_log_text(stderr_path)
            combined_text = stdout_text + "\n" + stderr_text

            real_percent = parse_snap_percent_from_text(combined_text)

            if real_percent is not None:
                progress_percent = real_percent
                source = "real SNAP percent"
                stage = last_stage
            else:
                estimated_percent, stage = estimate_snap_stage_percent(combined_text)
                progress_percent = estimated_percent
                source = "estimated stage"

            quiet_time_bonus = min(12.0, elapsed_minutes * 0.5)
            progress_percent = max(progress_percent, best_percent, quiet_time_bonus)

            if returncode is None:
                progress_percent = min(progress_percent, 97.0)
            else:
                if returncode == 0:
                    progress_percent = 100.0
                    stage = "completed"
                    source = "process complete"
                else:
                    progress_percent = max(progress_percent, best_percent)
                    stage = "failed"
                    source = f"return code {returncode}"

            best_percent = max(best_percent, progress_percent)
            last_stage = stage

            snap_output_size_mb = (
                file_size_mb(monitored_output_path)
                if monitored_output_path is not None
                else 0.0
            )

            print_progress_line(
                percent_value=progress_percent,
                elapsed_minutes=elapsed_minutes,
                stage=stage,
                snap_output_size_mb=snap_output_size_mb,
                source=source,
            )

            if returncode is not None:
                print()
                return int(returncode)

            time.sleep(max(2, int(progress_interval_seconds)))


def band_descriptions(src: rasterio.io.DatasetReader) -> list[str]:
    descs = []
    for i in range(1, src.count + 1):
        desc = src.descriptions[i - 1] or ""
        tags = src.tags(i)
        tag_desc = tags.get("DESCRIPTION", "") or tags.get("long_name", "") or ""
        name = f"{desc} {tag_desc}".strip()
        descs.append(name)
    return descs


def find_vv_vh_band_indexes(src: rasterio.io.DatasetReader) -> tuple[int, int, list[str]]:
    descs = band_descriptions(src)
    lower = [d.lower() for d in descs]

    vv_candidates = []
    vh_candidates = []

    for idx, text in enumerate(lower, start=1):
        if "vv" in text and "vh" not in text:
            vv_candidates.append(idx)
        if "vh" in text:
            vh_candidates.append(idx)

    if vv_candidates and vh_candidates:
        return vv_candidates[0], vh_candidates[0], descs

    if src.count >= 2:
        return 1, 2, descs

    raise ValueError(f"SNAP output has insufficient bands: {src.count}")


def make_output_profile(target: rasterio.io.DatasetReader) -> dict:
    blockx = min(512, target.width)
    blocky = min(512, target.height)

    blockx = max(16, (blockx // 16) * 16)
    blocky = max(16, (blocky // 16) * 16)

    return {
        "driver": "GTiff",
        "height": target.height,
        "width": target.width,
        "count": 3,
        "dtype": "float32",
        "crs": target.crs,
        "transform": target.transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": blockx,
        "blockysize": blocky,
        "BIGTIFF": "IF_SAFER",
        "nodata": None,
    }


def make_uint8_profile(target: rasterio.io.DatasetReader) -> dict:
    blockx = min(512, target.width)
    blocky = min(512, target.height)

    blockx = max(16, (blockx // 16) * 16)
    blocky = max(16, (blocky // 16) * 16)

    return {
        "driver": "GTiff",
        "height": target.height,
        "width": target.width,
        "count": 1,
        "dtype": "uint8",
        "crs": target.crs,
        "transform": target.transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": blockx,
        "blockysize": blocky,
        "BIGTIFF": "IF_SAFER",
        "nodata": None,
    }


def read_warped_band(
    src: rasterio.io.DatasetReader,
    band_index: int,
    target: rasterio.io.DatasetReader,
) -> tuple[np.ndarray, np.ndarray]:
    vrt_options = {
        "crs": target.crs,
        "transform": target.transform,
        "width": target.width,
        "height": target.height,
        "resampling": Resampling.bilinear,
        "dst_nodata": NODATA_FLOAT,
    }

    if src.nodata is not None:
        vrt_options["src_nodata"] = src.nodata

    with WarpedVRT(src, **vrt_options) as vrt:
        data = vrt.read(band_index, out_dtype="float32", masked=True)
        mask = vrt.read_masks(band_index) > 0

    if np.ma.isMaskedArray(data):
        arr = data.filled(NODATA_FLOAT).astype(np.float32)
        valid = (~np.ma.getmaskarray(data)) & mask
    else:
        arr = data.astype(np.float32)
        valid = mask

    valid &= np.isfinite(arr)
    valid &= arr != NODATA_FLOAT

    return arr, valid


def align_snap_output_to_target(
    snap_tif_path: Path,
    target_s2_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict:
    safe_unlink(output_path, overwrite=args.overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(target_s2_path) as target, rasterio.open(snap_tif_path) as src:
        if src.crs is None:
            raise ValueError(f"SNAP output has no CRS: {snap_tif_path}")

        vv_idx, vh_idx, descs = find_vv_vh_band_indexes(src)

        vv, vv_valid = read_warped_band(src, vv_idx, target)
        vh, vh_valid = read_warped_band(src, vh_idx, target)

        valid = vv_valid & vh_valid

        if args.fallback_all_zero_as_nodata:
            valid &= ~((vv == 0) & (vh == 0))

        vv_out = np.where(valid, vv, 0).astype(np.float32)
        vh_out = np.where(valid, vh, 0).astype(np.float32)
        diff_out = np.where(valid, vv - vh, 0).astype(np.float32)

        profile = make_output_profile(target)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(vv_out, 1)
            dst.write(vh_out, 2)
            dst.write(diff_out, 3)
            dst.write_mask(valid.astype(np.uint8) * 255)

            dst.set_band_description(1, "VV_dB")
            dst.set_band_description(2, "VH_dB")
            dst.set_band_description(3, "VV_minus_VH_dB")

            dst.update_tags(
                source_snap_output=str(snap_tif_path),
                source_snap_band_descriptions=" | ".join(descs),
                source_vv_band_index=str(vv_idx),
                source_vh_band_index=str(vh_idx),
                aligned_to_s2=str(target_s2_path),
                fallback_all_zero_as_nodata=str(args.fallback_all_zero_as_nodata),
            )

        total_pixels = target.width * target.height
        valid_pixels = int(valid.sum())

        return {
            "fallback_valid_pixels": valid_pixels,
            "fallback_valid_percent": percent(valid_pixels, total_pixels),
            "fallback_invalid_pixels": int(total_pixels - valid_pixels),
            "fallback_invalid_percent": percent(total_pixels - valid_pixels, total_pixels),
            "snap_source_band_descriptions": " | ".join(descs),
            "snap_vv_band_index": vv_idx,
            "snap_vh_band_index": vh_idx,
        }


def build_original_s1_missing_mask(
    original_s1_path: Path,
    all_zero_as_nodata: bool,
) -> tuple[np.ndarray, dict]:
    with rasterio.open(original_s1_path) as src:
        height = src.height
        width = src.width
        total = height * width

        combined = np.zeros((height, width), dtype=bool)
        official = np.zeros((height, width), dtype=bool)
        all_zero = np.zeros((height, width), dtype=bool)
        nonfinite = np.zeros((height, width), dtype=bool)

        indexes = list(range(1, src.count + 1))

        for _, window in src.block_windows(1):
            row0 = int(window.row_off)
            row1 = int(window.row_off + window.height)
            col0 = int(window.col_off)
            col1 = int(window.col_off + window.width)

            masks = src.read_masks(indexes=indexes, window=window)
            official_block = np.any(masks == 0, axis=0)
            block = official_block.copy()

            official[row0:row1, col0:col1] = official_block

            data = src.read(indexes=indexes, window=window)
            nonfinite_block = np.any(~np.isfinite(data), axis=0)
            nonfinite[row0:row1, col0:col1] = nonfinite_block
            block |= nonfinite_block

            if all_zero_as_nodata:
                zero_block = np.all(data == 0, axis=0)
                all_zero[row0:row1, col0:col1] = zero_block
                block |= zero_block

            combined[row0:row1, col0:col1] = block

        return combined, {
            "original_s1_total_pixels": total,
            "original_s1_missing_pixels": int(combined.sum()),
            "original_s1_missing_percent": percent(int(combined.sum()), total),
            "original_s1_official_nodata_pixels": int(official.sum()),
            "original_s1_all_zero_pixels": int(all_zero.sum()),
            "original_s1_nonfinite_pixels": int(nonfinite.sum()),
        }


def read_dataset_valid_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        masks = src.read_masks()
        return np.all(masks > 0, axis=0)


def write_mask_raster(
    mask: np.ndarray,
    target_s2_path: Path,
    output_path: Path,
    description: str,
    overwrite: bool,
) -> None:
    safe_unlink(output_path, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(target_s2_path) as target:
        profile = make_uint8_profile(target)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
            dst.write_mask(np.full(mask.shape, 255, dtype=np.uint8))
            dst.set_band_description(1, description)


def compute_fillability_qc(
    city: str,
    target_s2_path: Path,
    original_s1_path: Path,
    fallback_output_path: Path,
    qc_dir: Path,
    args: argparse.Namespace,
) -> dict:
    original_missing, original_meta = build_original_s1_missing_mask(
        original_s1_path=original_s1_path,
        all_zero_as_nodata=args.original_s1_all_zero_as_nodata,
    )

    fallback_valid = read_dataset_valid_mask(fallback_output_path)

    if fallback_valid.shape != original_missing.shape:
        raise ValueError(
            f"Shape mismatch between original S1 missing mask and fallback valid mask for {city}"
        )

    fillable = original_missing & fallback_valid
    expected_remaining = original_missing & ~fallback_valid

    masks_dir = qc_dir / "masks"

    fallback_valid_path = masks_dir / f"{city}_fallback_valid_mask.tif"
    original_missing_path = masks_dir / f"{city}_original_s1_missing_mask.tif"
    fillable_path = masks_dir / f"{city}_original_missing_fillable_by_fallback_mask.tif"

    write_mask_raster(
        mask=fallback_valid,
        target_s2_path=target_s2_path,
        output_path=fallback_valid_path,
        description="fallback S1 valid mask: 1=valid",
        overwrite=args.overwrite,
    )

    write_mask_raster(
        mask=original_missing,
        target_s2_path=target_s2_path,
        output_path=original_missing_path,
        description="original S1 missing mask: 1=missing",
        overwrite=args.overwrite,
    )

    write_mask_raster(
        mask=fillable,
        target_s2_path=target_s2_path,
        output_path=fillable_path,
        description="original S1 missing pixels fillable by fallback: 1=fillable",
        overwrite=args.overwrite,
    )

    total = original_missing.size
    original_missing_pixels = int(original_missing.sum())
    fillable_pixels = int(fillable.sum())
    remaining_pixels = int(expected_remaining.sum())

    return {
        **original_meta,
        "original_missing_fillable_by_fallback_pixels": fillable_pixels,
        "original_missing_fillable_percent_of_missing": percent(
            fillable_pixels, original_missing_pixels
        ),
        "original_missing_fillable_percent_of_raster": percent(fillable_pixels, total),
        "expected_remaining_original_s1_nodata_pixels_after_merge": remaining_pixels,
        "expected_remaining_original_s1_nodata_percent_after_merge": percent(
            remaining_pixels, total
        ),
        "fallback_valid_mask_path": str(fallback_valid_path),
        "original_s1_missing_mask_path": str(original_missing_path),
        "original_missing_fillable_by_fallback_mask_path": str(fillable_path),
    }


def process_city(
    row: dict,
    instance_root: Path,
    raw_root: Path,
    output_root: Path,
    tmp_root: Path,
    qc_dir: Path,
    args: argparse.Namespace,
) -> dict:
    city = str(row["city"])
    selected_item_id = str(row["selected_item_id"])

    target_s2_path = find_s2_target(instance_root, args.s2_subdir, city)
    original_s1_path = find_original_s1(instance_root, args.original_s1_subdir, city)
    safe_dir, manifest_path = find_safe_manifest(raw_root, city, selected_item_id)

    city_tmp = tmp_root / city / selected_item_id
    city_tmp.mkdir(parents=True, exist_ok=True)

    snap_output_path = city_tmp / f"{city}_fallback_snap_tc_db.tif"
    graph_path = qc_dir / "graphs" / f"{city}_fallback_s1_snap_graph.xml"
    stdout_path = qc_dir / "logs" / f"{city}_gpt_stdout.txt"
    stderr_path = qc_dir / "logs" / f"{city}_gpt_stderr.txt"

    output_path = output_root / city / f"{city}_s1_fallback_vv_vh_vvdiff_10m_aligned.tif"

    result = {
        "city": city,
        "selected_item_id": selected_item_id,
        "status": "",
        "message": "",
        "safe_dir": str(safe_dir),
        "manifest_path": str(manifest_path),
        "target_s2_path": str(target_s2_path),
        "original_s1_path": str(original_s1_path),
        "snap_output_path": str(snap_output_path),
        "output_path": str(output_path),
        "graph_path": str(graph_path),
        "gpt_stdout_path": str(stdout_path),
        "gpt_stderr_path": str(stderr_path),
    }

    try:
        with rasterio.open(target_s2_path) as target:
            graph_xml = build_snap_graph_xml(
                manifest_path=manifest_path,
                output_tif_path=snap_output_path,
                target_crs=target.crs,
                args=args,
            )

        graph_path.parent.mkdir(parents=True, exist_ok=True)
        if graph_path.exists() and not args.overwrite:
            raise FileExistsError(f"Graph exists. Use --overwrite: {graph_path}")
        graph_path.write_text(graph_xml, encoding="utf-8")

        if args.run_snap:
            if snap_output_path.exists() and args.overwrite:
                snap_output_path.unlink()

            gpt_path = Path(args.gpt_path)
            if not gpt_path.exists():
                raise FileNotFoundError(f"SNAP GPT executable does not exist: {gpt_path}")

            print(f"[INFO] Running SNAP GPT for {city}...")
            returncode = run_snap_gpt(
                gpt_path=gpt_path,
                graph_path=graph_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                snap_cache=args.snap_cache,
                progress_interval_seconds=args.progress_interval_seconds,
                monitored_output_path=snap_output_path,
            )
            result["gpt_returncode"] = returncode

            if returncode != 0:
                result["status"] = "error_snap_gpt_failed"
                result["message"] = f"SNAP GPT failed with return code {returncode}."
                return result
        else:
            result["gpt_returncode"] = "skipped"

        if not snap_output_path.exists():
            result["status"] = "error_snap_output_missing"
            result["message"] = f"SNAP output GeoTIFF not found: {snap_output_path}"
            return result

        align_stats = align_snap_output_to_target(
            snap_tif_path=snap_output_path,
            target_s2_path=target_s2_path,
            output_path=output_path,
            args=args,
        )

        fillability = compute_fillability_qc(
            city=city,
            target_s2_path=target_s2_path,
            original_s1_path=original_s1_path,
            fallback_output_path=output_path,
            qc_dir=qc_dir,
            args=args,
        )

        result.update(align_stats)
        result.update(fillability)

        if fillability["original_missing_fillable_percent_of_missing"] >= 95.0:
            result["status"] = "processed_fallback_ready_high_coverage"
            result["message"] = "Fallback S1 processed and covers most original missing S1 pixels."
        elif fillability["original_missing_fillable_percent_of_missing"] > 0:
            result["status"] = "processed_fallback_ready_partial_coverage"
            result["message"] = "Fallback S1 processed but only partially covers original missing pixels."
        else:
            result["status"] = "processed_fallback_no_missing_overlap"
            result["message"] = "Fallback S1 processed but does not cover original missing pixels."

        if not args.keep_temp:
            try:
                if city_tmp.exists():
                    shutil.rmtree(city_tmp)
            except Exception as exc:
                result["temp_cleanup_warning"] = str(exc)

        return result

    except Exception as exc:
        result["status"] = "error"
        result["message"] = str(exc)
        return result


def main() -> None:
    args = parse_args()

    instance_root = Path(args.instance_root)

    if not instance_root.exists():
        raise FileNotFoundError(f"Instance root does not exist: {instance_root}")

    selection_csv = (
        Path(args.selection_csv)
        if args.selection_csv
        else instance_root / "qc" / "s1_fallback_selection" / "selected_s1_fallback_products.csv"
    )

    raw_root = (
        Path(args.raw_root)
        if args.raw_root
        else instance_root / "fallback_s1" / "raw"
    )

    output_root = (
        Path(args.output_root)
        if args.output_root
        else instance_root / "fallback_s1" / "processed"
    )

    tmp_root = (
        Path(args.tmp_root)
        if args.tmp_root
        else instance_root / "fallback_s1" / "snap_tmp"
    )

    qc_dir = (
        Path(args.qc_dir)
        if args.qc_dir
        else instance_root / "qc" / "s1_fallback_preprocess"
    )

    output_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (qc_dir / "logs").mkdir(parents=True, exist_ok=True)
    (qc_dir / "masks").mkdir(parents=True, exist_ok=True)

    selection_df = load_selection_table(selection_csv, args.cities)

    print(f"[INFO] Instance root: {instance_root}")
    print(f"[INFO] Selection CSV: {selection_csv}")
    print(f"[INFO] Raw root: {raw_root}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Temp root: {tmp_root}")
    print(f"[INFO] QC dir: {qc_dir}")
    print(f"[INFO] Cities to process: {len(selection_df)}")
    print(f"[INFO] GPT path: {args.gpt_path}")
    print(f"[INFO] SNAP cache: {args.snap_cache}")
    print(f"[INFO] progress interval: {args.progress_interval_seconds}s")
    print(f"[INFO] run_snap: {args.run_snap}")
    print(f"[INFO] keep_temp: {args.keep_temp}")

    rows: list[dict] = []

    for idx, row in enumerate(selection_df.to_dict(orient="records"), start=1):
        city = str(row["city"])
        item_id = str(row["selected_item_id"])

        print(f"\n[STEP {idx}/{len(selection_df)}] {city}")
        print(f"[INFO] Selected item: {item_id}")

        result = process_city(
            row=row,
            instance_root=instance_root,
            raw_root=raw_root,
            output_root=output_root,
            tmp_root=tmp_root,
            qc_dir=qc_dir,
            args=args,
        )
        rows.append(result)

        print(
            "[OK] "
            f"status={result.get('status')} | "
            f"fallback_valid={result.get('fallback_valid_percent', '')} | "
            f"fillable={result.get('original_missing_fillable_percent_of_missing', '')} | "
            f"message={result.get('message', '')}"
        )

    csv_path = qc_dir / "s1_fallback_preprocess_summary.csv"
    json_path = qc_dir / "s1_fallback_preprocess_summary.json"
    md_path = qc_dir / "s1_fallback_preprocess_summary.md"

    write_csv(rows, csv_path, overwrite=args.overwrite)
    write_json(rows, csv_path.with_suffix(".json"), overwrite=args.overwrite)
    write_markdown(rows, md_path, overwrite=args.overwrite)

    print("\n[DONE] Wrote:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    print("\n[SUMMARY]")
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()