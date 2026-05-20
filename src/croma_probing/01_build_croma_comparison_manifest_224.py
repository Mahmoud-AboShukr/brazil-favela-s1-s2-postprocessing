#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_build_croma_comparison_manifest_224.py

Build the CROMA comparison manifest for Instance C 224x224 patches.

This script does NOT extract CROMA embeddings.
It creates the clean experiment manifest that later scripts will use.

Input:
    <instance-root>/metadata/instance_C_patches/
        patch_metadata_ps224_st112_cover.csv

Optional validation input:
    <instance-root>/metadata/instance_C_patches/
        patch_metadata_validation_ps224_st112_cover.json

Outputs:
    <instance-root>/metadata/croma_probing/
        croma_patch_manifest_ps224_st112_cover.csv
        croma_comparison_manifest_ps224_st112_cover.csv
        croma_modality_summary_ps224_st112_cover.csv
        croma_comparison_manifest_ps224_st112_cover.json
        croma_comparison_manifest_ps224_st112_cover.md

Main fair comparison contract:
    S2:
        12 optical bands

    SNAP-GRD:
        use only bands 1 and 2:
            1 = VV
            2 = VH
        ignore band 3:
            3 = VV_minus_VH

    RTC:
        use bands 1 and 2:
            1 = VV
            2 = VH

Primary modalities:
    s2
    s1_snap_vv_vh
    s1_rtc_vv_vh
    s2_s1_snap_vv_vh
    s2_s1_rtc_vv_vh

Example:

python src/croma_probing/01_build_croma_comparison_manifest_224.py `
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
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


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


def safe_int(value: object, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        fail(f"Input CSV does not exist: {path_to_str(path)}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        fail(f"Input CSV is empty: {path_to_str(path)}")

    return rows


def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    overwrite: bool,
    fieldnames: Optional[List[str]] = None,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    if fieldnames is None:
        if not rows:
            fail(f"No rows and no fieldnames provided for CSV: {path_to_str(path)}")
        fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json_optional(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_markdown(
    path: Path,
    summary: Dict[str, object],
    modality_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# CROMA comparison manifest")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Patch metadata CSV: `{summary['patch_metadata_csv']}`")
    lines.append(f"- Validation JSON: `{summary['validation_json']}`")
    lines.append(f"- Patch manifest CSV: `{summary['outputs']['patch_manifest_csv']}`")
    lines.append(f"- Comparison manifest CSV: `{summary['outputs']['comparison_manifest_csv']}`")
    lines.append(f"- Total patches: `{summary['total_patches']}`")
    lines.append(f"- Positive patches: `{summary['positive_patches']}`")
    lines.append(f"- Empty patches: `{summary['empty_patches']}`")
    lines.append(f"- Modalities: `{';'.join(summary['modalities'])}`")
    lines.append(f"- Total comparison rows: `{summary['total_comparison_rows']}`")
    lines.append("")

    lines.append("## Fair comparison contract")
    lines.append("")
    lines.append("The primary RTC-vs-SNAP-GRD comparison uses identical patch IDs and labels.")
    lines.append("")
    lines.append("| Source | Available bands | Used bands in primary CROMA comparison | Notes |")
    lines.append("|---|---:|---|---|")
    lines.append("| S2 | 12 | 1--12 | Optical branch |")
    lines.append("| SNAP-GRD | 3 | 1--2 = VV,VH | Band 3 VV_minus_VH is ignored for fairness |")
    lines.append("| RTC | 2 | 1--2 = VV,VH | Native CROMA-compatible SAR input |")
    lines.append("| Label | 1 | binary patch label | Same labels for all modalities |")
    lines.append("")

    lines.append("## Modality summary")
    lines.append("")
    lines.append(
        "| modality | rows | uses S2 | uses SAR | SAR variant | optical bands | SAR bands | positive rows | empty rows |"
    )
    lines.append("|---|---:|---|---|---|---|---|---:|---:|")

    for row in modality_rows:
        lines.append(
            f"| {row['modality']} | "
            f"{row['rows']} | "
            f"{row['uses_s2']} | "
            f"{row['uses_sar']} | "
            f"{row['sar_variant']} | "
            f"{row['optical_band_indices']} | "
            f"{row['sar_band_indices']} | "
            f"{row['positive_rows']} | "
            f"{row['empty_rows']} |"
        )

    lines.append("")
    lines.append("## Readiness checks inherited from patch metadata")
    lines.append("")
    for key, value in summary["readiness"].items():
        lines.append(f"- {key}: `{value}`")

    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("The next script should extract frozen CROMA embeddings for each manifest modality.")
    lines.append("The embedding extraction script should read `croma_comparison_manifest_ps224_st112_cover.csv` and write embeddings grouped by modality.")
    lines.append("For the primary fair comparison, the SNAP-GRD SAR input must use only VV/VH, not VV_minus_VH.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

REQUIRED_METADATA_COLUMNS = [
    "patch_id",
    "city",
    "region",
    "row_start",
    "col_start",
    "height",
    "width",
    "patch_size",
    "stride",
    "edge_mode",
    "city_width",
    "city_height",
    "source_s2_path",
    "source_s1_snap_path",
    "source_s1_rtc_path",
    "source_label_path",
    "s2_band_count",
    "s1_snap_band_count",
    "s1_rtc_band_count",
    "label_band_count",
    "s2_valid_percent",
    "s1_snap_valid_percent",
    "s1_rtc_valid_percent",
    "s1_rtc_zero_percent",
    "patch_label_binary",
    "label_positive_pixels",
    "label_positive_percent",
    "label_density_bin",
    "label_non_binary",
]


def validate_metadata_rows(
    rows: List[Dict[str, str]],
    *,
    args: argparse.Namespace,
) -> None:
    columns = set(rows[0].keys())
    missing = [col for col in REQUIRED_METADATA_COLUMNS if col not in columns]

    if missing:
        fail("Patch metadata is missing required columns:\n" + "\n".join(f"  - {c}" for c in missing))

    if len(rows) != int(args.expected_total_patches):
        fail(f"Expected {args.expected_total_patches} metadata rows, got {len(rows)}")

    patch_ids = [r["patch_id"] for r in rows]
    duplicates = [patch_id for patch_id, count in Counter(patch_ids).items() if count > 1]

    if duplicates:
        fail("Duplicate patch IDs found. Example:\n" + "\n".join(f"  - {p}" for p in duplicates[:20]))

    cities = sorted(set(normalize_city(r["city"]) for r in rows))

    if len(cities) != int(args.expected_city_count):
        fail(f"Expected {args.expected_city_count} cities, got {len(cities)}")

    bad_rows: List[str] = []

    for r in rows:
        patch_id = r["patch_id"]

        if safe_int(r["s2_band_count"]) != int(args.expected_s2_bands):
            bad_rows.append(f"{patch_id}: s2_band_count={r['s2_band_count']}")

        if safe_int(r["s1_snap_band_count"]) != int(args.expected_s1_snap_bands):
            bad_rows.append(f"{patch_id}: s1_snap_band_count={r['s1_snap_band_count']}")

        if safe_int(r["s1_rtc_band_count"]) != int(args.expected_s1_rtc_bands):
            bad_rows.append(f"{patch_id}: s1_rtc_band_count={r['s1_rtc_band_count']}")

        if safe_int(r["label_band_count"]) != int(args.expected_label_bands):
            bad_rows.append(f"{patch_id}: label_band_count={r['label_band_count']}")

        if safe_float(r["s2_valid_percent"]) < 100.0 - float(args.percent_tolerance):
            bad_rows.append(f"{patch_id}: s2_valid_percent={r['s2_valid_percent']}")

        if safe_float(r["s1_snap_valid_percent"]) < 100.0 - float(args.percent_tolerance):
            bad_rows.append(f"{patch_id}: s1_snap_valid_percent={r['s1_snap_valid_percent']}")

        if safe_float(r["s1_rtc_valid_percent"]) < 100.0 - float(args.percent_tolerance):
            bad_rows.append(f"{patch_id}: s1_rtc_valid_percent={r['s1_rtc_valid_percent']}")

        if safe_float(r["s1_rtc_zero_percent"]) > float(args.zero_percent_tolerance):
            bad_rows.append(f"{patch_id}: s1_rtc_zero_percent={r['s1_rtc_zero_percent']}")

        if parse_bool(r["label_non_binary"]):
            bad_rows.append(f"{patch_id}: label_non_binary={r['label_non_binary']}")

        positive_pixels = safe_int(r["label_positive_pixels"])
        patch_label_binary = safe_int(r["patch_label_binary"])
        expected_patch_label_binary = 1 if positive_pixels > 0 else 0

        if patch_label_binary != expected_patch_label_binary:
            bad_rows.append(
                f"{patch_id}: patch_label_binary={patch_label_binary}, "
                f"label_positive_pixels={positive_pixels}"
            )

    if bad_rows:
        fail(
            "Patch metadata failed strict readiness checks. First issues:\n"
            + "\n".join(f"  - {x}" for x in bad_rows[:50])
        )

    unique_paths = set()

    for r in rows:
        unique_paths.add(r["source_s2_path"])
        unique_paths.add(r["source_s1_snap_path"])
        unique_paths.add(r["source_s1_rtc_path"])
        unique_paths.add(r["source_label_path"])

    missing_paths = sorted(p for p in unique_paths if not Path(p).exists())

    if missing_paths:
        fail("Some source paths do not exist. First missing paths:\n" + "\n".join(f"  - {p}" for p in missing_paths[:50]))


def validate_metadata_validation_json(path: Path, args: argparse.Namespace) -> Dict[str, object]:
    payload = read_json_optional(path)

    if payload is None:
        if args.require_validation_json:
            fail(f"Validation JSON is required but does not exist: {path_to_str(path)}")
        return {}

    status = payload.get("validation_status", "")
    error_count = safe_int(payload.get("error_count", 999))
    warning_count = safe_int(payload.get("warning_count", 999))
    readiness = payload.get("readiness", {})

    if args.require_validation_json:
        if status != "passed":
            fail(f"Patch metadata validation JSON status is not passed: {status}")

        if error_count != 0:
            fail(f"Patch metadata validation JSON has error_count={error_count}")

        required_readiness = [
            "s2_valid_all_patches",
            "s1_snap_valid_all_patches",
            "s1_rtc_valid_all_patches",
            "s1_rtc_available_all_patches",
            "s1_rtc_zero_free_all_patches",
            "labels_binary",
            "source_paths_exist",
            "raster_stacks_aligned",
        ]

        failed = [key for key in required_readiness if readiness.get(key) is not True]

        if failed:
            fail("Patch metadata validation JSON readiness failed:\n" + "\n".join(f"  - {x}" for x in failed))

    return payload


# ---------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------

MODALITY_CONFIGS: Dict[str, Dict[str, object]] = {
    "s2": {
        "uses_s2": True,
        "uses_sar": False,
        "sar_variant": "none",
        "optical_band_indices": "1;2;3;4;5;6;7;8;9;10;11;12",
        "optical_channel_names": "B01;B02;B03;B04;B05;B06;B07;B08;B8A;B09;B11;B12",
        "sar_band_indices": "",
        "sar_channel_names": "",
        "snap_ignored_band_indices": "",
        "snap_ignored_channel_names": "",
    },
    "s1_snap_vv_vh": {
        "uses_s2": False,
        "uses_sar": True,
        "sar_variant": "snap_grd",
        "optical_band_indices": "",
        "optical_channel_names": "",
        "sar_band_indices": "1;2",
        "sar_channel_names": "VV;VH",
        "snap_ignored_band_indices": "3",
        "snap_ignored_channel_names": "VV_minus_VH",
    },
    "s1_rtc_vv_vh": {
        "uses_s2": False,
        "uses_sar": True,
        "sar_variant": "rtc",
        "optical_band_indices": "",
        "optical_channel_names": "",
        "sar_band_indices": "1;2",
        "sar_channel_names": "VV;VH",
        "snap_ignored_band_indices": "",
        "snap_ignored_channel_names": "",
    },
    "s2_s1_snap_vv_vh": {
        "uses_s2": True,
        "uses_sar": True,
        "sar_variant": "snap_grd",
        "optical_band_indices": "1;2;3;4;5;6;7;8;9;10;11;12",
        "optical_channel_names": "B01;B02;B03;B04;B05;B06;B07;B08;B8A;B09;B11;B12",
        "sar_band_indices": "1;2",
        "sar_channel_names": "VV;VH",
        "snap_ignored_band_indices": "3",
        "snap_ignored_channel_names": "VV_minus_VH",
    },
    "s2_s1_rtc_vv_vh": {
        "uses_s2": True,
        "uses_sar": True,
        "sar_variant": "rtc",
        "optical_band_indices": "1;2;3;4;5;6;7;8;9;10;11;12",
        "optical_channel_names": "B01;B02;B03;B04;B05;B06;B07;B08;B8A;B09;B11;B12",
        "sar_band_indices": "1;2",
        "sar_channel_names": "VV;VH",
        "snap_ignored_band_indices": "",
        "snap_ignored_channel_names": "",
    },
}


def build_patch_manifest_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    patch_rows: List[Dict[str, object]] = []

    for r in rows:
        patch_rows.append(
            {
                "patch_id": r["patch_id"],
                "city": normalize_city(r["city"]),
                "region": r.get("region", ""),
                "row_start": safe_int(r["row_start"]),
                "col_start": safe_int(r["col_start"]),
                "height": safe_int(r["height"]),
                "width": safe_int(r["width"]),
                "patch_size": safe_int(r["patch_size"]),
                "stride": safe_int(r["stride"]),
                "edge_mode": r.get("edge_mode", ""),
                "city_width": safe_int(r["city_width"]),
                "city_height": safe_int(r["city_height"]),
                "label_binary": safe_int(r["patch_label_binary"]),
                "label_positive_pixels": safe_int(r["label_positive_pixels"]),
                "label_positive_percent": safe_float(r["label_positive_percent"]),
                "label_density_bin": r["label_density_bin"],
                "source_s2_path": r["source_s2_path"],
                "source_s1_snap_path": r["source_s1_snap_path"],
                "source_s1_rtc_path": r["source_s1_rtc_path"],
                "source_label_path": r["source_label_path"],
                "s2_band_count": safe_int(r["s2_band_count"]),
                "s1_snap_band_count": safe_int(r["s1_snap_band_count"]),
                "s1_rtc_band_count": safe_int(r["s1_rtc_band_count"]),
                "label_band_count": safe_int(r["label_band_count"]),
            }
        )

    return patch_rows


def build_comparison_manifest_rows(
    rows: List[Dict[str, str]],
    modalities: Sequence[str],
) -> List[Dict[str, object]]:
    comparison_rows: List[Dict[str, object]] = []

    unknown = [m for m in modalities if m not in MODALITY_CONFIGS]
    if unknown:
        fail("Unknown modalities requested:\n" + "\n".join(f"  - {m}" for m in unknown))

    for r in rows:
        patch_id = r["patch_id"]

        for modality in modalities:
            cfg = MODALITY_CONFIGS[modality]

            uses_s2 = bool(cfg["uses_s2"])
            uses_sar = bool(cfg["uses_sar"])
            sar_variant = str(cfg["sar_variant"])

            if sar_variant == "snap_grd":
                sar_path = r["source_s1_snap_path"]
                sar_available_band_count = safe_int(r["s1_snap_band_count"])
                sar_expected_available_band_count = 3
                sar_source_name = "S1_SNAP_GRD"
            elif sar_variant == "rtc":
                sar_path = r["source_s1_rtc_path"]
                sar_available_band_count = safe_int(r["s1_rtc_band_count"])
                sar_expected_available_band_count = 2
                sar_source_name = "S1_RTC"
            else:
                sar_path = ""
                sar_available_band_count = 0
                sar_expected_available_band_count = 0
                sar_source_name = ""

            optical_path = r["source_s2_path"] if uses_s2 else ""
            optical_available_band_count = safe_int(r["s2_band_count"]) if uses_s2 else 0

            manifest_row_id = f"{patch_id}__{modality}"

            comparison_rows.append(
                {
                    "manifest_row_id": manifest_row_id,
                    "patch_id": patch_id,
                    "modality": modality,
                    "city": normalize_city(r["city"]),
                    "region": r.get("region", ""),
                    "row_start": safe_int(r["row_start"]),
                    "col_start": safe_int(r["col_start"]),
                    "height": safe_int(r["height"]),
                    "width": safe_int(r["width"]),
                    "patch_size": safe_int(r["patch_size"]),
                    "stride": safe_int(r["stride"]),
                    "edge_mode": r.get("edge_mode", ""),
                    "city_width": safe_int(r["city_width"]),
                    "city_height": safe_int(r["city_height"]),

                    "label_binary": safe_int(r["patch_label_binary"]),
                    "label_positive_pixels": safe_int(r["label_positive_pixels"]),
                    "label_positive_percent": safe_float(r["label_positive_percent"]),
                    "label_density_bin": r["label_density_bin"],
                    "label_path": r["source_label_path"],
                    "label_band_indices": "1",

                    "uses_s2": uses_s2,
                    "uses_sar": uses_sar,
                    "sar_variant": sar_variant,
                    "sar_source_name": sar_source_name,

                    "optical_path": optical_path,
                    "optical_available_band_count": optical_available_band_count,
                    "optical_band_indices": cfg["optical_band_indices"],
                    "optical_channel_names": cfg["optical_channel_names"],

                    "sar_path": sar_path,
                    "sar_available_band_count": sar_available_band_count,
                    "sar_expected_available_band_count": sar_expected_available_band_count,
                    "sar_band_indices": cfg["sar_band_indices"],
                    "sar_channel_names": cfg["sar_channel_names"],

                    "snap_ignored_band_indices": cfg["snap_ignored_band_indices"],
                    "snap_ignored_channel_names": cfg["snap_ignored_channel_names"],

                    "fair_primary_comparison": True,
                    "notes": (
                        "SNAP-GRD band 3 VV_minus_VH intentionally ignored."
                        if sar_variant == "snap_grd"
                        else ""
                    ),
                }
            )

    return comparison_rows


def build_modality_summary_rows(
    comparison_rows: List[Dict[str, object]],
    modalities: Sequence[str],
) -> List[Dict[str, object]]:
    rows_by_modality: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for row in comparison_rows:
        rows_by_modality[str(row["modality"])].append(row)

    summary_rows: List[Dict[str, object]] = []

    for modality in modalities:
        rows = rows_by_modality[modality]
        if not rows:
            continue

        first = rows[0]
        positive = sum(1 for r in rows if safe_int(r["label_binary"]) == 1)
        empty = len(rows) - positive

        summary_rows.append(
            {
                "modality": modality,
                "rows": len(rows),
                "uses_s2": first["uses_s2"],
                "uses_sar": first["uses_sar"],
                "sar_variant": first["sar_variant"],
                "optical_band_indices": first["optical_band_indices"],
                "optical_channel_names": first["optical_channel_names"],
                "sar_band_indices": first["sar_band_indices"],
                "sar_channel_names": first["sar_channel_names"],
                "snap_ignored_band_indices": first["snap_ignored_band_indices"],
                "snap_ignored_channel_names": first["snap_ignored_channel_names"],
                "positive_rows": positive,
                "empty_rows": empty,
                "positive_percent": round(100.0 * positive / len(rows), 8) if rows else 0.0,
            }
        )

    return summary_rows


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    *,
    instance_root: Path,
    patch_metadata_csv: Path,
    validation_json_path: Path,
    validation_payload: Dict[str, object],
    patch_rows: List[Dict[str, object]],
    comparison_rows: List[Dict[str, object]],
    modality_rows: List[Dict[str, object]],
    modalities: Sequence[str],
    args: argparse.Namespace,
    output_paths: Dict[str, Path],
) -> Dict[str, object]:
    total_patches = len(patch_rows)
    positive_patches = sum(1 for r in patch_rows if safe_int(r["label_binary"]) == 1)
    empty_patches = total_patches - positive_patches

    density_counter = Counter(str(r["label_density_bin"]) for r in patch_rows)

    readiness = validation_payload.get("readiness", {}) if validation_payload else {
        "s2_valid_all_patches": True,
        "s1_snap_valid_all_patches": True,
        "s1_rtc_valid_all_patches": True,
        "s1_rtc_available_all_patches": True,
        "s1_rtc_zero_free_all_patches": True,
        "labels_binary": True,
        "source_paths_exist": True,
        "raster_stacks_aligned": True,
    }

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "patch_metadata_csv": path_to_str(patch_metadata_csv),
        "validation_json": path_to_str(validation_json_path),
        "total_patches": total_patches,
        "positive_patches": positive_patches,
        "empty_patches": empty_patches,
        "positive_patch_percent": round(100.0 * positive_patches / total_patches, 8) if total_patches else 0.0,
        "city_count": len(set(str(r["city"]) for r in patch_rows)),
        "modalities": list(modalities),
        "n_modalities": len(modalities),
        "total_comparison_rows": len(comparison_rows),
        "expected_comparison_rows": total_patches * len(modalities),
        "label_density_bins": {
            "empty": density_counter.get("empty", 0),
            "low": density_counter.get("low", 0),
            "medium": density_counter.get("medium", 0),
            "high": density_counter.get("high", 0),
        },
        "readiness": readiness,
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "expected_total_patches": args.expected_total_patches,
            "expected_city_count": args.expected_city_count,
            "expected_positive_patches": args.expected_positive_patches,
            "expected_empty_patches": args.expected_empty_patches,
            "expected_s2_bands": args.expected_s2_bands,
            "expected_s1_snap_bands": args.expected_s1_snap_bands,
            "expected_s1_rtc_bands": args.expected_s1_rtc_bands,
            "require_validation_json": bool(args.require_validation_json),
            "modalities": list(modalities),
        },
        "outputs": {key: path_to_str(value) for key, value in output_paths.items()},
        "modality_summary": modality_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CROMA comparison manifest for Instance C 224x224 patches."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--patch-metadata-csv",
        type=Path,
        default=None,
        help="Patch metadata CSV. Default: <instance-root>/metadata/instance_C_patches/patch_metadata_ps<patch-size>_st<stride>_<edge-mode>.csv",
    )

    parser.add_argument(
        "--validation-json",
        type=Path,
        default=None,
        help="Patch metadata validation JSON. Default: <instance-root>/metadata/instance_C_patches/patch_metadata_validation_ps<patch-size>_st<stride>_<edge-mode>.json",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <instance-root>/metadata/croma_probing.",
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
        help="Stride. Default: 112.",
    )

    parser.add_argument(
        "--edge-mode",
        choices=["cover", "drop"],
        default="cover",
        help="Edge mode. Default: cover.",
    )

    parser.add_argument(
        "--modalities",
        nargs="+",
        default=[
            "s2",
            "s1_snap_vv_vh",
            "s1_rtc_vv_vh",
            "s2_s1_snap_vv_vh",
            "s2_s1_rtc_vv_vh",
        ],
        help="Modalities to include in the long comparison manifest.",
    )

    parser.add_argument(
        "--expected-total-patches",
        type=int,
        default=12699,
        help="Expected total patch count. Default: 12699.",
    )

    parser.add_argument(
        "--expected-city-count",
        type=int,
        default=26,
        help="Expected city count. Default: 26.",
    )

    parser.add_argument(
        "--expected-positive-patches",
        type=int,
        default=6382,
        help="Expected positive patch count. Default: 6382.",
    )

    parser.add_argument(
        "--expected-empty-patches",
        type=int,
        default=6317,
        help="Expected empty patch count. Default: 6317.",
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
        help="Expected S1 SNAP-GRD available band count. Default: 3.",
    )

    parser.add_argument(
        "--expected-s1-rtc-bands",
        type=int,
        default=2,
        help="Expected S1 RTC available band count. Default: 2.",
    )

    parser.add_argument(
        "--expected-label-bands",
        type=int,
        default=1,
        help="Expected label band count. Default: 1.",
    )

    parser.add_argument(
        "--percent-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for 100 percent checks. Default: 1e-6.",
    )

    parser.add_argument(
        "--zero-percent-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for RTC zero percent checks. Default: 1e-6.",
    )

    parser.add_argument(
        "--require-validation-json",
        action="store_true",
        default=True,
        help="Require the patch metadata validation JSON to exist and have passed. Default: enabled.",
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
    metadata_dir = instance_root / "metadata" / "instance_C_patches"
    output_dir: Path = args.output_dir or (instance_root / "metadata" / "croma_probing")

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    patch_metadata_csv: Path = args.patch_metadata_csv or (
        metadata_dir / f"patch_metadata_{stem}.csv"
    )

    validation_json_path: Path = args.validation_json or (
        metadata_dir / f"patch_metadata_validation_{stem}.json"
    )

    patch_manifest_csv = output_dir / f"croma_patch_manifest_{stem}.csv"
    comparison_manifest_csv = output_dir / f"croma_comparison_manifest_{stem}.csv"
    modality_summary_csv = output_dir / f"croma_modality_summary_{stem}.csv"
    json_path = output_dir / f"croma_comparison_manifest_{stem}.json"
    md_path = output_dir / f"croma_comparison_manifest_{stem}.md"

    output_paths = {
        "patch_manifest_csv": patch_manifest_csv,
        "comparison_manifest_csv": comparison_manifest_csv,
        "modality_summary_csv": modality_summary_csv,
        "json": json_path,
        "markdown": md_path,
    }

    log("STEP", "Building CROMA comparison manifest.")
    log("INFO", f"Instance root:      {path_to_str(instance_root)}")
    log("INFO", f"Patch metadata CSV: {path_to_str(patch_metadata_csv)}")
    log("INFO", f"Validation JSON:    {path_to_str(validation_json_path)}")
    log("INFO", f"Output dir:         {path_to_str(output_dir)}")
    log("INFO", f"Modalities:         {';'.join(args.modalities)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    validation_payload = validate_metadata_validation_json(validation_json_path, args)

    metadata_rows = read_csv_rows(patch_metadata_csv)
    validate_metadata_rows(metadata_rows, args=args)

    patch_rows = build_patch_manifest_rows(metadata_rows)
    comparison_rows = build_comparison_manifest_rows(metadata_rows, args.modalities)
    modality_rows = build_modality_summary_rows(comparison_rows, args.modalities)

    expected_comparison_rows = len(metadata_rows) * len(args.modalities)

    if len(comparison_rows) != expected_comparison_rows:
        fail(
            f"Comparison manifest row count mismatch: "
            f"got {len(comparison_rows)}, expected {expected_comparison_rows}"
        )

    summary = build_summary(
        instance_root=instance_root,
        patch_metadata_csv=patch_metadata_csv,
        validation_json_path=validation_json_path,
        validation_payload=validation_payload,
        patch_rows=patch_rows,
        comparison_rows=comparison_rows,
        modality_rows=modality_rows,
        modalities=args.modalities,
        args=args,
        output_paths=output_paths,
    )

    log("STEP", "Writing outputs.")

    write_csv(patch_manifest_csv, patch_rows, overwrite=bool(args.overwrite))
    write_csv(comparison_manifest_csv, comparison_rows, overwrite=bool(args.overwrite))
    write_csv(modality_summary_csv, modality_rows, overwrite=bool(args.overwrite))
    write_json(json_path, summary, overwrite=bool(args.overwrite))
    write_markdown(md_path, summary, modality_rows, overwrite=bool(args.overwrite))

    log("OK", f"Wrote patch manifest:      {path_to_str(patch_manifest_csv)}")
    log("OK", f"Wrote comparison manifest: {path_to_str(comparison_manifest_csv)}")
    log("OK", f"Wrote modality summary:    {path_to_str(modality_summary_csv)}")
    log("OK", f"Wrote JSON:                {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:            {path_to_str(md_path)}")

    log("STEP", "Final summary.")
    log("OK", f"Total patches: {summary['total_patches']}")
    log("OK", f"Positive patches: {summary['positive_patches']}")
    log("OK", f"Empty patches: {summary['empty_patches']}")
    log("OK", f"Modalities: {summary['n_modalities']}")
    log("OK", f"Total comparison rows: {summary['total_comparison_rows']}")
    log("OK", f"Expected comparison rows: {summary['expected_comparison_rows']}")

    for row in modality_rows:
        log(
            "OK",
            f"{row['modality']}: rows={row['rows']}, "
            f"positive={row['positive_rows']}, empty={row['empty_rows']}, "
            f"sar_variant={row['sar_variant']}, sar_bands={row['sar_band_indices']}",
        )


if __name__ == "__main__":
    main()