#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_extract_croma_embeddings_smoke_test_224.py

Real-data CROMA embedding extraction smoke test for Instance C.

This script does NOT run full embedding extraction.
It tests the full real-data path on a small balanced subset:

    - read the CROMA comparison manifest;
    - select a small balanced set of positive and empty patches;
    - load real raster windows from S2, SNAP-GRD, and RTC;
    - use only VV/VH for both SNAP-GRD and RTC;
    - normalize inputs in a CROMA-compatible way;
    - run official PretrainedCROMA;
    - save small .npz embedding files;
    - write CSV/JSON/Markdown summaries.

Primary fair comparison contract:

    S2:
        12 channels

    SNAP-GRD:
        available bands = 3
        used bands = 1,2 = VV,VH
        ignored band = 3 = VV_minus_VH

    RTC:
        available bands = 2
        used bands = 1,2 = VV,VH

CROMA output used as primary embedding:

    s2:
        optical_GAP

    s1_snap_vv_vh:
        SAR_GAP

    s1_rtc_vv_vh:
        SAR_GAP

    s2_s1_snap_vv_vh:
        joint_GAP

    s2_s1_rtc_vv_vh:
        joint_GAP

Example:

python src/croma_probing/03_extract_croma_embeddings_smoke_test_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --croma-repo "C:/Users/acer/OneDrive/Desktop/UMR_espace_dev/CROMA" `
  --weights-path "D:/models/CROMA/CROMA_base.pt" `
  --model-size base `
  --image-resolution 224 `
  --positive-samples 5 `
  --empty-samples 5 `
  --batch-size 1 `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import random
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
except ImportError as exc:
    raise SystemExit(
        "[ERROR] rasterio is required.\n"
        "Install it with:\n"
        "    pip install rasterio\n\n"
        f"Original error: {exc}"
    )

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "[ERROR] torch is required.\n"
        "Install PyTorch first.\n\n"
        f"Original error: {exc}"
    )


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


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_city(value: str) -> str:
    value = str(value).strip()
    value = value.replace("\\", "/").split("/")[-1]
    value = value.lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


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
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


# ---------------------------------------------------------------------
# CSV / JSON / Markdown
# ---------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        fail(f"CSV does not exist: {path_to_str(path)}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        fail(f"CSV is empty: {path_to_str(path)}")

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
            fail(f"No rows and no fieldnames for CSV: {path_to_str(path)}")
        fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(
    path: Path,
    summary: Dict[str, object],
    selected_rows: List[Dict[str, object]],
    extraction_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# CROMA real-data embedding smoke test")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Comparison manifest: `{summary['comparison_manifest_csv']}`")
    lines.append(f"- CROMA repo: `{summary['croma_repo']}`")
    lines.append(f"- Weights path: `{summary['weights_path']}`")
    lines.append(f"- Model size: `{summary['parameters']['model_size']}`")
    lines.append(f"- Image resolution: `{summary['parameters']['image_resolution']}`")
    lines.append(f"- Device: `{summary['device']}`")
    lines.append(f"- Selected patch count: `{summary['selected_patch_count']}`")
    lines.append(f"- Modalities: `{';'.join(summary['modalities'])}`")
    lines.append(f"- Error count: `{summary['error_count']}`")
    lines.append("")

    lines.append("## Selected patches")
    lines.append("")
    lines.append("| patch_id | city | region | label | label_positive_percent | density_bin |")
    lines.append("|---|---|---|---:|---:|---|")
    for row in selected_rows:
        lines.append(
            f"| {row['patch_id']} | "
            f"{row['city']} | "
            f"{row['region']} | "
            f"{row['label_binary']} | "
            f"{row['label_positive_percent']} | "
            f"{row['label_density_bin']} |"
        )

    lines.append("")
    lines.append("## Extraction results")
    lines.append("")
    lines.append(
        "| modality | status | rows | embedding key | embedding shape | output NPZ | notes |"
    )
    lines.append("|---|---|---:|---|---|---|---|")
    for row in extraction_rows:
        lines.append(
            f"| {row['modality']} | "
            f"{row['status']} | "
            f"{row['n_rows']} | "
            f"{row['embedding_key']} | "
            f"{row['embedding_shape']} | "
            f"`{row['output_npz']}` | "
            f"{row['notes']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- This is a smoke test only; it verifies real raster loading and CROMA inference on a small balanced subset.")
    lines.append("- SNAP-GRD uses only VV/VH bands 1 and 2. Its VV-minus-VH band is intentionally ignored.")
    lines.append("- RTC uses VV/VH bands 1 and 2.")
    lines.append("- For full extraction, use the same loading and normalization logic but run over all manifest rows.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# CROMA setup
# ---------------------------------------------------------------------

def import_pretrained_croma(croma_repo: Path):
    if not croma_repo.exists():
        fail(f"CROMA repo path does not exist: {path_to_str(croma_repo)}")

    use_croma_path = croma_repo / "use_croma.py"

    if not use_croma_path.exists():
        fail(f"use_croma.py not found: {path_to_str(use_croma_path)}")

    repo_str = str(croma_repo.resolve())

    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    try:
        from use_croma import PretrainedCROMA
    except Exception as exc:
        fail(f"Could not import PretrainedCROMA from use_croma.py: {repr(exc)}")

    return PretrainedCROMA


def choose_device(device_index: int, force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        if device_index < 0 or device_index >= torch.cuda.device_count():
            fail(f"Requested CUDA device {device_index}, but device_count={torch.cuda.device_count()}")
        return torch.device(f"cuda:{device_index}")

    log("WARN", "CUDA is not available. Falling back to CPU.")
    return torch.device("cpu")


def clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Manifest selection
# ---------------------------------------------------------------------

def expected_modalities_default() -> List[str]:
    return [
        "s2",
        "s1_snap_vv_vh",
        "s1_rtc_vv_vh",
        "s2_s1_snap_vv_vh",
        "s2_s1_rtc_vv_vh",
    ]


def unique_patch_rows_from_manifest(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    patch_rows: List[Dict[str, str]] = []

    for row in rows:
        patch_id = row["patch_id"]

        if patch_id in seen:
            continue

        seen.add(patch_id)
        patch_rows.append(row)

    return patch_rows


def select_balanced_patch_ids(
    rows: List[Dict[str, str]],
    *,
    positive_samples: int,
    empty_samples: int,
    seed: int,
) -> List[str]:
    unique_rows = unique_patch_rows_from_manifest(rows)

    positive = [r for r in unique_rows if safe_int(r["label_binary"]) == 1]
    empty = [r for r in unique_rows if safe_int(r["label_binary"]) == 0]

    if len(positive) < positive_samples:
        fail(f"Requested {positive_samples} positive samples, but only {len(positive)} are available.")

    if len(empty) < empty_samples:
        fail(f"Requested {empty_samples} empty samples, but only {len(empty)} are available.")

    rng = random.Random(seed)

    positive_sorted = sorted(positive, key=lambda r: r["patch_id"])
    empty_sorted = sorted(empty, key=lambda r: r["patch_id"])

    selected_positive = rng.sample(positive_sorted, positive_samples)
    selected_empty = rng.sample(empty_sorted, empty_samples)

    selected = selected_positive + selected_empty
    selected = sorted(selected, key=lambda r: (safe_int(r["label_binary"]), r["city"], r["patch_id"]))

    return [r["patch_id"] for r in selected]


def build_selected_patch_summary(
    rows: List[Dict[str, str]],
    selected_patch_ids: Sequence[str],
) -> List[Dict[str, object]]:
    patch_map: Dict[str, Dict[str, str]] = {}

    for row in rows:
        if row["patch_id"] not in patch_map:
            patch_map[row["patch_id"]] = row

    selected_rows: List[Dict[str, object]] = []

    for patch_id in selected_patch_ids:
        row = patch_map[patch_id]

        selected_rows.append(
            {
                "patch_id": patch_id,
                "city": normalize_city(row["city"]),
                "region": row.get("region", ""),
                "label_binary": safe_int(row["label_binary"]),
                "label_positive_pixels": safe_int(row["label_positive_pixels"]),
                "label_positive_percent": safe_float(row["label_positive_percent"]),
                "label_density_bin": row["label_density_bin"],
            }
        )

    return selected_rows


def rows_for_modality_and_patches(
    rows: List[Dict[str, str]],
    modality: str,
    selected_patch_ids: Sequence[str],
) -> List[Dict[str, str]]:
    selected_set = set(selected_patch_ids)

    filtered = [
        row for row in rows
        if row["modality"] == modality and row["patch_id"] in selected_set
    ]

    order = {patch_id: idx for idx, patch_id in enumerate(selected_patch_ids)}
    filtered.sort(key=lambda r: order[r["patch_id"]])

    if len(filtered) != len(selected_patch_ids):
        missing = selected_set - set(r["patch_id"] for r in filtered)
        fail(
            f"Modality {modality} has {len(filtered)} rows, expected {len(selected_patch_ids)}. "
            f"Missing: {sorted(missing)[:10]}"
        )

    return filtered


# ---------------------------------------------------------------------
# Raster loading and normalization
# ---------------------------------------------------------------------

def parse_band_indices(text: str) -> List[int]:
    text = str(text).strip()

    if text == "":
        return []

    return [int(x) for x in text.split(";") if str(x).strip()]


def read_window_as_numpy(
    path: Path,
    *,
    band_indices: Sequence[int],
    row_start: int,
    col_start: int,
    height: int,
    width: int,
) -> np.ndarray:
    if not path.exists():
        fail(f"Raster path does not exist: {path_to_str(path)}")

    window = Window(
        col_off=int(col_start),
        row_off=int(row_start),
        width=int(width),
        height=int(height),
    )

    with rasterio.open(path) as src:
        arr = src.read(list(band_indices), window=window, masked=False).astype(np.float32)

    if arr.shape[1] != int(height) or arr.shape[2] != int(width):
        fail(
            f"Unexpected raster window shape from {path_to_str(path)}: "
            f"{arr.shape}, expected ({len(band_indices)}, {height}, {width})"
        )

    if not np.isfinite(arr).all():
        fail(f"Non-finite values found in raster window: {path_to_str(path)}")

    return arr


def normalize_like_croma_readme(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Batch/channel normalization inspired by the official CROMA README.

    For each channel:
        min = mean - 2 * std
        max = mean + 2 * std
        scaled = clip((x - min) / (max - min), 0, 1)

    This works for S2 reflectance-like values and SAR dB-like values.
    """

    x = x.float()
    normalized_channels: List[torch.Tensor] = []

    for channel_idx in range(x.shape[1]):
        channel = x[:, channel_idx, :, :]
        mean = channel.mean()
        std = channel.std(unbiased=False)

        min_value = mean - 2.0 * std
        max_value = mean + 2.0 * std
        denom = torch.clamp(max_value - min_value, min=eps)

        normalized = (channel - min_value) / denom
        normalized = torch.clamp(normalized, 0.0, 1.0)
        normalized_channels.append(normalized.unsqueeze(1))

    return torch.cat(normalized_channels, dim=1)


def build_batch_tensors(
    rows: List[Dict[str, str]],
    *,
    device: torch.device,
    normalize_inputs: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[Dict[str, object]]]:
    optical_arrays: List[np.ndarray] = []
    sar_arrays: List[np.ndarray] = []
    sample_rows: List[Dict[str, object]] = []

    for row in rows:
        uses_s2 = parse_bool(row["uses_s2"])
        uses_sar = parse_bool(row["uses_sar"])

        row_start = safe_int(row["row_start"])
        col_start = safe_int(row["col_start"])
        height = safe_int(row["height"])
        width = safe_int(row["width"])

        if uses_s2:
            optical_band_indices = parse_band_indices(row["optical_band_indices"])
            optical = read_window_as_numpy(
                Path(row["optical_path"]),
                band_indices=optical_band_indices,
                row_start=row_start,
                col_start=col_start,
                height=height,
                width=width,
            )

            if optical.shape[0] != 12:
                fail(f"{row['manifest_row_id']}: expected 12 optical channels, got {optical.shape[0]}")

            optical_arrays.append(optical)

        if uses_sar:
            sar_band_indices = parse_band_indices(row["sar_band_indices"])
            sar = read_window_as_numpy(
                Path(row["sar_path"]),
                band_indices=sar_band_indices,
                row_start=row_start,
                col_start=col_start,
                height=height,
                width=width,
            )

            if sar.shape[0] != 2:
                fail(f"{row['manifest_row_id']}: expected 2 SAR channels, got {sar.shape[0]}")

            sar_arrays.append(sar)

        sample_rows.append(
            {
                "manifest_row_id": row["manifest_row_id"],
                "patch_id": row["patch_id"],
                "modality": row["modality"],
                "city": normalize_city(row["city"]),
                "region": row.get("region", ""),
                "label_binary": safe_int(row["label_binary"]),
                "label_positive_pixels": safe_int(row["label_positive_pixels"]),
                "label_positive_percent": safe_float(row["label_positive_percent"]),
                "label_density_bin": row["label_density_bin"],
            }
        )

    optical_tensor: Optional[torch.Tensor] = None
    sar_tensor: Optional[torch.Tensor] = None

    if optical_arrays:
        optical_np = np.stack(optical_arrays, axis=0).astype(np.float32)
        optical_tensor = torch.from_numpy(optical_np).to(device)
        if normalize_inputs:
            optical_tensor = normalize_like_croma_readme(optical_tensor)

    if sar_arrays:
        sar_np = np.stack(sar_arrays, axis=0).astype(np.float32)
        sar_tensor = torch.from_numpy(sar_np).to(device)
        if normalize_inputs:
            sar_tensor = normalize_like_croma_readme(sar_tensor)

    return sar_tensor, optical_tensor, sample_rows


# ---------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------

def croma_modality_for_manifest_modality(modality: str) -> str:
    if modality == "s2":
        return "optical"

    if modality in {"s1_snap_vv_vh", "s1_rtc_vv_vh"}:
        return "SAR"

    if modality in {"s2_s1_snap_vv_vh", "s2_s1_rtc_vv_vh"}:
        return "both"

    fail(f"Unsupported modality: {modality}")


def primary_embedding_key_for_modality(modality: str) -> str:
    if modality == "s2":
        return "optical_GAP"

    if modality in {"s1_snap_vv_vh", "s1_rtc_vv_vh"}:
        return "SAR_GAP"

    if modality in {"s2_s1_snap_vv_vh", "s2_s1_rtc_vv_vh"}:
        return "joint_GAP"

    fail(f"Unsupported modality: {modality}")


def output_keys_for_model_modality(model_modality: str) -> List[str]:
    if model_modality == "optical":
        return ["optical_GAP", "optical_encodings"]

    if model_modality == "SAR":
        return ["SAR_GAP", "SAR_encodings"]

    if model_modality == "both":
        return [
            "SAR_GAP",
            "SAR_encodings",
            "optical_GAP",
            "optical_encodings",
            "joint_GAP",
            "joint_encodings",
        ]

    fail(f"Unsupported CROMA model modality: {model_modality}")


def extract_for_modality(
    *,
    modality: str,
    rows: List[Dict[str, str]],
    PretrainedCROMA,
    weights_path: Path,
    model_size: str,
    image_resolution: int,
    device: torch.device,
    batch_size: int,
    normalize_inputs: bool,
    output_dir: Path,
    output_stem: str,
    overwrite: bool,
) -> Dict[str, object]:
    log("STEP", f"Extracting smoke-test embeddings for modality: {modality}")

    model_modality = croma_modality_for_manifest_modality(modality)
    primary_key = primary_embedding_key_for_modality(modality)

    output_npz = output_dir / f"croma_smoke_embeddings_{modality}_{output_stem}.npz"

    ensure_output_can_be_written(output_npz, overwrite)

    model = None

    all_embeddings: List[np.ndarray] = []
    all_patch_ids: List[str] = []
    all_manifest_row_ids: List[str] = []
    all_cities: List[str] = []
    all_regions: List[str] = []
    all_labels: List[int] = []
    all_label_percents: List[float] = []

    output_shapes: Dict[str, str] = {}

    try:
        model = PretrainedCROMA(
            pretrained_path=str(weights_path),
            size=str(model_size),
            modality=model_modality,
            image_resolution=int(image_resolution),
        ).to(device)

        model.eval()

        expected_output_keys = output_keys_for_model_modality(model_modality)

        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start:start + batch_size]

                sar_tensor, optical_tensor, sample_rows = build_batch_tensors(
                    batch_rows,
                    device=device,
                    normalize_inputs=normalize_inputs,
                )

                kwargs = {}

                if sar_tensor is not None:
                    kwargs["SAR_images"] = sar_tensor

                if optical_tensor is not None:
                    kwargs["optical_images"] = optical_tensor

                outputs = model(**kwargs)

                for key in expected_output_keys:
                    if key not in outputs:
                        fail(
                            f"Expected output key `{key}` missing for modality {modality}. "
                            f"Actual keys: {sorted(outputs.keys())}"
                        )

                    output_shapes[key] = str(tuple(outputs[key].shape))

                embedding = outputs[primary_key].detach().float().cpu().numpy()

                if embedding.ndim != 2:
                    fail(f"Primary embedding {primary_key} should be 2D [N,D], got shape {embedding.shape}")

                all_embeddings.append(embedding)

                for sample in sample_rows:
                    all_manifest_row_ids.append(str(sample["manifest_row_id"]))
                    all_patch_ids.append(str(sample["patch_id"]))
                    all_cities.append(str(sample["city"]))
                    all_regions.append(str(sample["region"]))
                    all_labels.append(int(sample["label_binary"]))
                    all_label_percents.append(float(sample["label_positive_percent"]))

        embeddings = np.concatenate(all_embeddings, axis=0)

        np.savez_compressed(
            output_npz,
            embeddings=embeddings.astype(np.float32),
            patch_ids=np.array(all_patch_ids, dtype=object),
            manifest_row_ids=np.array(all_manifest_row_ids, dtype=object),
            cities=np.array(all_cities, dtype=object),
            regions=np.array(all_regions, dtype=object),
            label_binary=np.array(all_labels, dtype=np.int64),
            label_positive_percent=np.array(all_label_percents, dtype=np.float32),
            modality=np.array([modality], dtype=object),
            croma_model_modality=np.array([model_modality], dtype=object),
            embedding_key=np.array([primary_key], dtype=object),
            output_shapes_json=np.array([json.dumps(output_shapes)], dtype=object),
        )

        return {
            "modality": modality,
            "status": "passed",
            "n_rows": len(rows),
            "embedding_key": primary_key,
            "embedding_shape": str(tuple(embeddings.shape)),
            "output_npz": path_to_str(output_npz),
            "croma_model_modality": model_modality,
            "output_shapes": json.dumps(output_shapes),
            "notes": "",
        }

    except Exception as exc:
        return {
            "modality": modality,
            "status": "failed",
            "n_rows": len(rows),
            "embedding_key": primary_key,
            "embedding_shape": "",
            "output_npz": path_to_str(output_npz),
            "croma_model_modality": model_modality,
            "output_shapes": json.dumps(output_shapes),
            "notes": traceback.format_exc().replace("\n", " | ")[:2000],
        }

    finally:
        if model is not None:
            del model
        clear_torch_memory()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    *,
    instance_root: Path,
    comparison_manifest_csv: Path,
    croma_repo: Path,
    weights_path: Path,
    selected_patch_rows: List[Dict[str, object]],
    extraction_rows: List[Dict[str, object]],
    args: argparse.Namespace,
    output_paths: Dict[str, Path],
    device: torch.device,
) -> Dict[str, object]:
    error_count = sum(1 for row in extraction_rows if row["status"] != "passed")

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if error_count == 0 else "failed",
        "error_count": error_count,
        "instance_root": path_to_str(instance_root),
        "comparison_manifest_csv": path_to_str(comparison_manifest_csv),
        "croma_repo": path_to_str(croma_repo),
        "weights_path": path_to_str(weights_path),
        "device": str(device),
        "selected_patch_count": len(selected_patch_rows),
        "selected_positive_patches": sum(1 for r in selected_patch_rows if safe_int(r["label_binary"]) == 1),
        "selected_empty_patches": sum(1 for r in selected_patch_rows if safe_int(r["label_binary"]) == 0),
        "modalities": list(args.modalities),
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "model_size": args.model_size,
            "image_resolution": args.image_resolution,
            "positive_samples": args.positive_samples,
            "empty_samples": args.empty_samples,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "normalize_inputs": bool(args.normalize_inputs),
            "device_index": args.device_index,
            "force_cpu": bool(args.force_cpu),
        },
        "outputs": {key: path_to_str(value) for key, value in output_paths.items()},
        "selected_patches": selected_patch_rows,
        "extraction_rows": extraction_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real-data CROMA embedding extraction smoke test."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--comparison-manifest-csv",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/croma_comparison_manifest_ps<patch-size>_st<stride>_<edge-mode>.csv",
    )

    parser.add_argument(
        "--croma-repo",
        type=Path,
        required=True,
        help="Path to official CROMA repo containing use_croma.py.",
    )

    parser.add_argument(
        "--weights-path",
        type=Path,
        required=True,
        help="Path to CROMA_base.pt or CROMA_large.pt.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/smoke_test.",
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
        default=expected_modalities_default(),
        help="Modalities to test.",
    )

    parser.add_argument(
        "--model-size",
        choices=["base", "large"],
        default="base",
        help="CROMA model size. Default: base.",
    )

    parser.add_argument(
        "--image-resolution",
        type=int,
        default=224,
        help="CROMA image resolution. Must match patch size and be multiple of 8. Default: 224.",
    )

    parser.add_argument(
        "--positive-samples",
        type=int,
        default=5,
        help="Number of positive patches for smoke test. Default: 5.",
    )

    parser.add_argument(
        "--empty-samples",
        type=int,
        default=5,
        help="Number of empty patches for smoke test. Default: 5.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patch sampling. Default: 42.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size. Default: 1 to reduce GPU memory pressure.",
    )

    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index. Default: 0.",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution.",
    )

    parser.add_argument(
        "--normalize-inputs",
        action="store_true",
        default=True,
        help="Normalize inputs using CROMA README-style per-batch channel scaling. Default: enabled.",
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
    output_dir: Path = args.output_dir or (instance_root / "metadata" / "croma_probing" / "smoke_test")

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    comparison_manifest_csv: Path = args.comparison_manifest_csv or (
        instance_root / "metadata" / "croma_probing" / f"croma_comparison_manifest_{stem}.csv"
    )

    selected_patches_csv = output_dir / f"croma_smoke_selected_patches_{stem}.csv"
    extraction_summary_csv = output_dir / f"croma_smoke_extraction_summary_{stem}.csv"
    json_path = output_dir / f"croma_smoke_test_{stem}.json"
    md_path = output_dir / f"croma_smoke_test_{stem}.md"

    output_paths = {
        "selected_patches_csv": selected_patches_csv,
        "extraction_summary_csv": extraction_summary_csv,
        "json": json_path,
        "markdown": md_path,
        "output_dir": output_dir,
    }

    log("STEP", "Running real-data CROMA embedding smoke test.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Manifest CSV:  {path_to_str(comparison_manifest_csv)}")
    log("INFO", f"CROMA repo:    {path_to_str(args.croma_repo)}")
    log("INFO", f"Weights path:  {path_to_str(args.weights_path)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not comparison_manifest_csv.exists():
        fail(f"Comparison manifest CSV does not exist: {path_to_str(comparison_manifest_csv)}")

    if not args.weights_path.exists():
        fail(f"CROMA weights path does not exist: {path_to_str(args.weights_path)}")

    if int(args.image_resolution) != int(args.patch_size):
        fail(
            f"image_resolution={args.image_resolution} must equal patch_size={args.patch_size} "
            "for this smoke test."
        )

    if int(args.image_resolution) % 8 != 0:
        fail("CROMA image_resolution must be divisible by 8.")

    PretrainedCROMA = import_pretrained_croma(args.croma_repo)
    device = choose_device(int(args.device_index), bool(args.force_cpu))

    log("INFO", f"Selected device: {device}")

    manifest_rows = read_csv_rows(comparison_manifest_csv)

    selected_patch_ids = select_balanced_patch_ids(
        manifest_rows,
        positive_samples=int(args.positive_samples),
        empty_samples=int(args.empty_samples),
        seed=int(args.seed),
    )

    selected_patch_rows = build_selected_patch_summary(manifest_rows, selected_patch_ids)

    log(
        "OK",
        f"Selected {len(selected_patch_ids)} patches "
        f"({args.positive_samples} positive + {args.empty_samples} empty).",
    )

    extraction_rows: List[Dict[str, object]] = []

    for modality in args.modalities:
        modality_rows = rows_for_modality_and_patches(
            manifest_rows,
            modality=modality,
            selected_patch_ids=selected_patch_ids,
        )

        result = extract_for_modality(
            modality=modality,
            rows=modality_rows,
            PretrainedCROMA=PretrainedCROMA,
            weights_path=args.weights_path,
            model_size=str(args.model_size),
            image_resolution=int(args.image_resolution),
            device=device,
            batch_size=int(args.batch_size),
            normalize_inputs=bool(args.normalize_inputs),
            output_dir=output_dir,
            output_stem=stem,
            overwrite=bool(args.overwrite),
        )

        extraction_rows.append(result)

        log(
            "OK" if result["status"] == "passed" else "ERROR",
            f"{modality}: status={result['status']}, "
            f"embedding_key={result['embedding_key']}, "
            f"shape={result['embedding_shape']}",
        )

    summary = build_summary(
        instance_root=instance_root,
        comparison_manifest_csv=comparison_manifest_csv,
        croma_repo=args.croma_repo,
        weights_path=args.weights_path,
        selected_patch_rows=selected_patch_rows,
        extraction_rows=extraction_rows,
        args=args,
        output_paths=output_paths,
        device=device,
    )

    log("STEP", "Writing smoke-test outputs.")

    write_csv(
        selected_patches_csv,
        selected_patch_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "patch_id",
            "city",
            "region",
            "label_binary",
            "label_positive_pixels",
            "label_positive_percent",
            "label_density_bin",
        ],
    )

    write_csv(
        extraction_summary_csv,
        extraction_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "modality",
            "status",
            "n_rows",
            "embedding_key",
            "embedding_shape",
            "output_npz",
            "croma_model_modality",
            "output_shapes",
            "notes",
        ],
    )

    write_json(json_path, summary, overwrite=bool(args.overwrite))
    write_markdown(md_path, summary, selected_patch_rows, extraction_rows, overwrite=bool(args.overwrite))

    log("OK", f"Wrote selected patches CSV: {path_to_str(selected_patches_csv)}")
    log("OK", f"Wrote extraction CSV:       {path_to_str(extraction_summary_csv)}")
    log("OK", f"Wrote JSON:                 {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:             {path_to_str(md_path)}")

    log("STEP", "Final smoke-test summary.")
    log("OK" if summary["status"] == "passed" else "ERROR", f"Status: {summary['status']}")
    log("OK", f"Selected patches: {summary['selected_patch_count']}")
    log("OK", f"Selected positives: {summary['selected_positive_patches']}")
    log("OK", f"Selected empty: {summary['selected_empty_patches']}")
    log("OK", f"Error count: {summary['error_count']}")

    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()