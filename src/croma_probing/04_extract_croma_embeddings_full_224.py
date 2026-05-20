#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_extract_croma_embeddings_full_224.py

Full CROMA embedding extraction for Instance C 224x224 patches.

This script extracts frozen CROMA embeddings for the validated comparison manifest:

    metadata/croma_probing/croma_comparison_manifest_ps224_st112_cover.csv

It supports the five primary fair-comparison modalities:

    s2
    s1_snap_vv_vh
    s1_rtc_vv_vh
    s2_s1_snap_vv_vh
    s2_s1_rtc_vv_vh

Fair comparison contract:

    S2:
        use 12 optical bands

    SNAP-GRD:
        available bands = 3
        use only bands 1 and 2 = VV,VH
        ignore band 3 = VV_minus_VH

    RTC:
        available bands = 2
        use bands 1 and 2 = VV,VH

Primary embedding saved per modality:

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

Why this is separate from the smoke test:
    - The smoke test used only 10 patches.
    - This script runs over all 12,699 patches per modality.
    - It writes one .npz file per modality.
    - It supports skip/resume at modality level.
    - It uses batch-independent per-sample/channel normalization by default.

Recommended first run: unimodal embeddings only

python src/croma_probing/04_extract_croma_embeddings_full_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --croma-repo "C:/Users/acer/OneDrive/Desktop/UMR_espace_dev/CROMA" `
  --weights-path "D:/models/CROMA/CROMA_base.pt" `
  --modalities s2 s1_snap_vv_vh s1_rtc_vv_vh `
  --model-size base `
  --image-resolution 224 `
  --batch-size 2 `
  --overwrite-summary

Recommended second run: joint embeddings

python src/croma_probing/04_extract_croma_embeddings_full_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --croma-repo "C:/Users/acer/OneDrive/Desktop/UMR_espace_dev/CROMA" `
  --weights-path "D:/models/CROMA/CROMA_base.pt" `
  --modalities s2_s1_snap_vv_vh s2_s1_rtc_vv_vh `
  --model-size base `
  --image-resolution 224 `
  --batch-size 1 `
  --overwrite-summary
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
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

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


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
            "Output already exists and overwrite was not enabled:\n"
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


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    modality_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Full CROMA embedding extraction")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Finished UTC: `{summary['finished_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Comparison manifest: `{summary['comparison_manifest_csv']}`")
    lines.append(f"- CROMA repo: `{summary['croma_repo']}`")
    lines.append(f"- Weights path: `{summary['weights_path']}`")
    lines.append(f"- Device: `{summary['device']}`")
    lines.append(f"- Model size: `{summary['parameters']['model_size']}`")
    lines.append(f"- Image resolution: `{summary['parameters']['image_resolution']}`")
    lines.append(f"- Batch size: `{summary['parameters']['batch_size']}`")
    lines.append(f"- Normalization: `{summary['parameters']['normalization']}`")
    lines.append(f"- Modalities requested: `{';'.join(summary['modalities_requested'])}`")
    lines.append(f"- Modalities completed: `{summary['modalities_completed']}`")
    lines.append(f"- Modalities skipped: `{summary['modalities_skipped']}`")
    lines.append(f"- Modalities failed: `{summary['modalities_failed']}`")
    lines.append("")

    lines.append("## Extraction results")
    lines.append("")
    lines.append(
        "| modality | status | rows | embedding key | embedding shape | output NPZ | seconds | notes |"
    )
    lines.append("|---|---|---:|---|---|---|---:|---|")

    for row in modality_rows:
        lines.append(
            f"| {row['modality']} | "
            f"{row['status']} | "
            f"{row['n_rows']} | "
            f"{row['embedding_key']} | "
            f"{row['embedding_shape']} | "
            f"`{row['output_npz']}` | "
            f"{row['elapsed_seconds']} | "
            f"{row['notes']} |"
        )

    lines.append("")
    lines.append("## Fair comparison contract")
    lines.append("")
    lines.append("- SNAP-GRD uses only bands 1 and 2: VV and VH.")
    lines.append("- SNAP-GRD band 3, VV-minus-VH, is ignored in the primary fair comparison.")
    lines.append("- RTC uses bands 1 and 2: VV and VH.")
    lines.append("- S2 uses all 12 optical bands.")
    lines.append("- Each modality output contains the same patch order as its manifest rows.")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("After all required modality embeddings are extracted, run frozen-probe experiments:")
    lines.append("")
    lines.append("```text")
    lines.append("src/croma_probing/05_train_frozen_probe_224.py")
    lines.append("```")

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
# Manifest and modality helpers
# ---------------------------------------------------------------------

def default_modalities() -> List[str]:
    return [
        "s2",
        "s1_snap_vv_vh",
        "s1_rtc_vv_vh",
        "s2_s1_snap_vv_vh",
        "s2_s1_rtc_vv_vh",
    ]


def validate_requested_modalities(modalities: Sequence[str]) -> None:
    allowed = set(default_modalities())
    unknown = [m for m in modalities if m not in allowed]

    if unknown:
        fail(
            "Unknown modalities requested:\n"
            + "\n".join(f"  - {m}" for m in unknown)
            + "\nAllowed modalities:\n"
            + "\n".join(f"  - {m}" for m in sorted(allowed))
        )


def croma_model_modality_for_manifest_modality(modality: str) -> str:
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


def expected_output_keys_for_model_modality(model_modality: str) -> List[str]:
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


def validate_manifest_rows(rows: List[Dict[str, str]]) -> None:
    required_columns = [
        "manifest_row_id",
        "patch_id",
        "modality",
        "city",
        "region",
        "row_start",
        "col_start",
        "height",
        "width",
        "label_binary",
        "label_positive_pixels",
        "label_positive_percent",
        "label_density_bin",
        "uses_s2",
        "uses_sar",
        "sar_variant",
        "optical_path",
        "optical_band_indices",
        "sar_path",
        "sar_band_indices",
        "sar_channel_names",
        "snap_ignored_band_indices",
    ]

    columns = set(rows[0].keys())
    missing = [col for col in required_columns if col not in columns]

    if missing:
        fail("Comparison manifest is missing columns:\n" + "\n".join(f"  - {c}" for c in missing))

    duplicate_manifest_rows = [
        row_id for row_id, count in Counter(r["manifest_row_id"] for r in rows).items()
        if count > 1
    ]

    if duplicate_manifest_rows:
        fail(
            "Duplicate manifest_row_id values found. First examples:\n"
            + "\n".join(f"  - {x}" for x in duplicate_manifest_rows[:20])
        )

    snap_bad = [
        r for r in rows
        if r["sar_variant"] == "snap_grd"
        and not (
            r["sar_band_indices"] == "1;2"
            and r["sar_channel_names"] == "VV;VH"
            and r["snap_ignored_band_indices"] == "3"
        )
    ]

    if snap_bad:
        fail(
            "Some SNAP-GRD rows do not use the fair VV/VH-only contract. "
            f"Bad rows: {len(snap_bad)}"
        )

    rtc_bad = [
        r for r in rows
        if r["sar_variant"] == "rtc"
        and not (
            r["sar_band_indices"] == "1;2"
            and r["sar_channel_names"] == "VV;VH"
        )
    ]

    if rtc_bad:
        fail(
            "Some RTC rows do not use VV/VH bands 1;2. "
            f"Bad rows: {len(rtc_bad)}"
        )


def rows_for_modality(rows: List[Dict[str, str]], modality: str, max_patches: Optional[int]) -> List[Dict[str, str]]:
    selected = [row for row in rows if row["modality"] == modality]
    selected.sort(key=lambda r: r["patch_id"])

    if max_patches is not None and max_patches > 0:
        selected = selected[:max_patches]

    if not selected:
        fail(f"No rows found for modality: {modality}")

    return selected


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

    expected_shape = (len(band_indices), int(height), int(width))

    if arr.shape != expected_shape:
        fail(
            f"Unexpected raster window shape from {path_to_str(path)}: "
            f"{arr.shape}, expected {expected_shape}"
        )

    if not np.isfinite(arr).all():
        fail(f"Non-finite values found in raster window: {path_to_str(path)}")

    return arr


def normalize_tensor(
    x: torch.Tensor,
    *,
    mode: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Normalize input tensor [B,C,H,W].

    Default mode:
        per_sample_channel_clip

    This is batch-independent:
        for each sample and channel, compute mean/std over H,W,
        then clip to [mean - 2 std, mean + 2 std] and scale to [0,1].

    This avoids embeddings depending on the batch composition.
    """

    x = x.float()

    if mode == "none":
        return x

    if mode == "per_sample_channel_clip":
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False)
        min_value = mean - 2.0 * std
        max_value = mean + 2.0 * std
        denom = torch.clamp(max_value - min_value, min=eps)
        return torch.clamp((x - min_value) / denom, 0.0, 1.0)

    if mode == "per_batch_channel_clip":
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True, unbiased=False)
        min_value = mean - 2.0 * std
        max_value = mean + 2.0 * std
        denom = torch.clamp(max_value - min_value, min=eps)
        return torch.clamp((x - min_value) / denom, 0.0, 1.0)

    fail(f"Unsupported normalization mode: {mode}")


def build_batch_tensors(
    rows: List[Dict[str, str]],
    *,
    device: torch.device,
    normalization: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, np.ndarray]]:
    optical_arrays: List[np.ndarray] = []
    sar_arrays: List[np.ndarray] = []

    manifest_row_ids: List[str] = []
    patch_ids: List[str] = []
    cities: List[str] = []
    regions: List[str] = []
    labels: List[int] = []
    label_positive_pixels: List[int] = []
    label_positive_percent: List[float] = []
    label_density_bins: List[str] = []

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

        manifest_row_ids.append(row["manifest_row_id"])
        patch_ids.append(row["patch_id"])
        cities.append(normalize_city(row["city"]))
        regions.append(row.get("region", ""))
        labels.append(safe_int(row["label_binary"]))
        label_positive_pixels.append(safe_int(row["label_positive_pixels"]))
        label_positive_percent.append(safe_float(row["label_positive_percent"]))
        label_density_bins.append(row["label_density_bin"])

    optical_tensor: Optional[torch.Tensor] = None
    sar_tensor: Optional[torch.Tensor] = None

    if optical_arrays:
        optical_np = np.stack(optical_arrays, axis=0).astype(np.float32)
        optical_tensor = torch.from_numpy(optical_np).to(device, non_blocking=True)
        optical_tensor = normalize_tensor(optical_tensor, mode=normalization)

    if sar_arrays:
        sar_np = np.stack(sar_arrays, axis=0).astype(np.float32)
        sar_tensor = torch.from_numpy(sar_np).to(device, non_blocking=True)
        sar_tensor = normalize_tensor(sar_tensor, mode=normalization)

    metadata = {
        "manifest_row_ids": np.asarray(manifest_row_ids, dtype=str),
        "patch_ids": np.asarray(patch_ids, dtype=str),
        "cities": np.asarray(cities, dtype=str),
        "regions": np.asarray(regions, dtype=str),
        "label_binary": np.asarray(labels, dtype=np.int64),
        "label_positive_pixels": np.asarray(label_positive_pixels, dtype=np.int64),
        "label_positive_percent": np.asarray(label_positive_percent, dtype=np.float32),
        "label_density_bins": np.asarray(label_density_bins, dtype=str),
    }

    return sar_tensor, optical_tensor, metadata


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------

def save_npz_atomic(path: Path, overwrite: bool, **arrays) -> None:
    ensure_output_can_be_written(path, overwrite)

    tmp_path = path.with_name(path.stem + f".tmp_{os.getpid()}" + path.suffix)

    if tmp_path.exists():
        tmp_path.unlink()

    try:
        np.savez_compressed(tmp_path, **arrays)

        # np.savez_compressed adds .npz if the path does not end with .npz.
        actual_tmp = tmp_path
        if not actual_tmp.exists() and Path(str(tmp_path) + ".npz").exists():
            actual_tmp = Path(str(tmp_path) + ".npz")

        if path.exists():
            path.unlink()

        actual_tmp.replace(path)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        if Path(str(tmp_path) + ".npz").exists():
            Path(str(tmp_path) + ".npz").unlink()
        raise


def extract_modality(
    *,
    modality: str,
    rows: List[Dict[str, str]],
    PretrainedCROMA,
    weights_path: Path,
    model_size: str,
    image_resolution: int,
    device: torch.device,
    batch_size: int,
    normalization: str,
    output_dir: Path,
    output_stem: str,
    overwrite_embeddings: bool,
    skip_existing: bool,
    progress_every: int,
) -> Dict[str, object]:
    started = time.time()

    model_modality = croma_model_modality_for_manifest_modality(modality)
    primary_key = primary_embedding_key_for_modality(modality)
    expected_keys = expected_output_keys_for_model_modality(model_modality)

    output_npz = output_dir / f"croma_embeddings_{modality}_{output_stem}.npz"

    if output_npz.exists() and skip_existing and not overwrite_embeddings:
        try:
            with np.load(output_npz, allow_pickle=False) as data:
                embedding_shape = tuple(data["embeddings"].shape)
                n_rows = int(data["embeddings"].shape[0])

            return {
                "modality": modality,
                "status": "skipped_existing",
                "n_rows": n_rows,
                "embedding_key": primary_key,
                "embedding_shape": str(embedding_shape),
                "output_npz": path_to_str(output_npz),
                "croma_model_modality": model_modality,
                "output_shapes": "",
                "elapsed_seconds": round(time.time() - started, 3),
                "notes": "Existing embedding file reused.",
            }

        except Exception as exc:
            log("WARN", f"Existing output could not be read and will be regenerated: {repr(exc)}")

    ensure_output_can_be_written(output_npz, overwrite_embeddings)

    log("STEP", f"Extracting modality: {modality}")
    log("INFO", f"Rows: {len(rows)}")
    log("INFO", f"CROMA model modality: {model_modality}")
    log("INFO", f"Primary embedding key: {primary_key}")
    log("INFO", f"Output: {path_to_str(output_npz)}")

    model = None

    all_embeddings: List[np.ndarray] = []
    all_manifest_row_ids: List[np.ndarray] = []
    all_patch_ids: List[np.ndarray] = []
    all_cities: List[np.ndarray] = []
    all_regions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_label_positive_pixels: List[np.ndarray] = []
    all_label_positive_percent: List[np.ndarray] = []
    all_label_density_bins: List[np.ndarray] = []

    output_shapes: Dict[str, str] = {}

    try:
        model = PretrainedCROMA(
            pretrained_path=str(weights_path),
            size=str(model_size),
            modality=model_modality,
            image_resolution=int(image_resolution),
        ).to(device)

        model.eval()

        progress = None

        if tqdm is not None:
            progress = tqdm(total=len(rows), desc=f"CROMA {modality}", unit="patch")

        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start:start + batch_size]

                sar_tensor, optical_tensor, metadata = build_batch_tensors(
                    batch_rows,
                    device=device,
                    normalization=normalization,
                )

                kwargs = {}

                if sar_tensor is not None:
                    kwargs["SAR_images"] = sar_tensor

                if optical_tensor is not None:
                    kwargs["optical_images"] = optical_tensor

                outputs = model(**kwargs)

                for key in expected_keys:
                    if key not in outputs:
                        fail(
                            f"Expected output key `{key}` missing for modality {modality}. "
                            f"Actual keys: {sorted(outputs.keys())}"
                        )

                    output_shapes[key] = str(tuple(outputs[key].shape))

                embedding = outputs[primary_key].detach().float().cpu().numpy()

                if embedding.ndim != 2:
                    fail(f"Primary embedding {primary_key} should be 2D [N,D], got {embedding.shape}")

                all_embeddings.append(embedding.astype(np.float32))
                all_manifest_row_ids.append(metadata["manifest_row_ids"])
                all_patch_ids.append(metadata["patch_ids"])
                all_cities.append(metadata["cities"])
                all_regions.append(metadata["regions"])
                all_labels.append(metadata["label_binary"])
                all_label_positive_pixels.append(metadata["label_positive_pixels"])
                all_label_positive_percent.append(metadata["label_positive_percent"])
                all_label_density_bins.append(metadata["label_density_bins"])

                if progress is not None:
                    progress.update(len(batch_rows))
                elif progress_every > 0 and (start + len(batch_rows)) % progress_every == 0:
                    log("INFO", f"{modality}: processed {start + len(batch_rows)}/{len(rows)} patches")

                del outputs
                del kwargs
                del sar_tensor
                del optical_tensor

        if progress is not None:
            progress.close()

        embeddings = np.concatenate(all_embeddings, axis=0)
        manifest_row_ids = np.concatenate(all_manifest_row_ids, axis=0)
        patch_ids = np.concatenate(all_patch_ids, axis=0)
        cities = np.concatenate(all_cities, axis=0)
        regions = np.concatenate(all_regions, axis=0)
        label_binary = np.concatenate(all_labels, axis=0)
        label_positive_pixels = np.concatenate(all_label_positive_pixels, axis=0)
        label_positive_percent = np.concatenate(all_label_positive_percent, axis=0)
        label_density_bins = np.concatenate(all_label_density_bins, axis=0)

        if embeddings.shape[0] != len(rows):
            fail(f"Embedding row mismatch for {modality}: {embeddings.shape[0]} != {len(rows)}")

        save_npz_atomic(
            output_npz,
            overwrite=overwrite_embeddings,
            embeddings=embeddings.astype(np.float32),
            manifest_row_ids=manifest_row_ids,
            patch_ids=patch_ids,
            cities=cities,
            regions=regions,
            label_binary=label_binary,
            label_positive_pixels=label_positive_pixels,
            label_positive_percent=label_positive_percent,
            label_density_bins=label_density_bins,
            modality=np.asarray([modality], dtype=str),
            croma_model_modality=np.asarray([model_modality], dtype=str),
            embedding_key=np.asarray([primary_key], dtype=str),
            output_shapes_json=np.asarray([json.dumps(output_shapes)], dtype=str),
            normalization=np.asarray([normalization], dtype=str),
            model_size=np.asarray([model_size], dtype=str),
            image_resolution=np.asarray([image_resolution], dtype=np.int64),
            created_utc=np.asarray([now_utc()], dtype=str),
        )

        elapsed = round(time.time() - started, 3)

        return {
            "modality": modality,
            "status": "completed",
            "n_rows": int(embeddings.shape[0]),
            "embedding_key": primary_key,
            "embedding_shape": str(tuple(embeddings.shape)),
            "output_npz": path_to_str(output_npz),
            "croma_model_modality": model_modality,
            "output_shapes": json.dumps(output_shapes),
            "elapsed_seconds": elapsed,
            "notes": "",
        }

    except RuntimeError as exc:
        message = str(exc)

        if "out of memory" in message.lower():
            notes = (
                "CUDA out of memory. Retry with --batch-size 1, "
                "or run fewer modalities at a time."
            )
        else:
            notes = traceback.format_exc().replace("\n", " | ")[:3000]

        return {
            "modality": modality,
            "status": "failed",
            "n_rows": len(rows),
            "embedding_key": primary_key,
            "embedding_shape": "",
            "output_npz": path_to_str(output_npz),
            "croma_model_modality": model_modality,
            "output_shapes": json.dumps(output_shapes),
            "elapsed_seconds": round(time.time() - started, 3),
            "notes": notes,
        }

    except Exception:
        return {
            "modality": modality,
            "status": "failed",
            "n_rows": len(rows),
            "embedding_key": primary_key,
            "embedding_shape": "",
            "output_npz": path_to_str(output_npz),
            "croma_model_modality": model_modality,
            "output_shapes": json.dumps(output_shapes),
            "elapsed_seconds": round(time.time() - started, 3),
            "notes": traceback.format_exc().replace("\n", " | ")[:3000],
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
    device: torch.device,
    modality_rows: List[Dict[str, object]],
    args: argparse.Namespace,
    output_paths: Dict[str, Path],
    started_utc: str,
) -> Dict[str, object]:
    completed = sum(1 for row in modality_rows if row["status"] == "completed")
    skipped = sum(1 for row in modality_rows if row["status"] == "skipped_existing")
    failed = sum(1 for row in modality_rows if row["status"] == "failed")

    status = "passed" if failed == 0 else "failed"

    return {
        "created_utc": started_utc,
        "finished_utc": now_utc(),
        "status": status,
        "instance_root": path_to_str(instance_root),
        "comparison_manifest_csv": path_to_str(comparison_manifest_csv),
        "croma_repo": path_to_str(croma_repo),
        "weights_path": path_to_str(weights_path),
        "device": str(device),
        "modalities_requested": list(args.modalities),
        "modalities_completed": completed,
        "modalities_skipped": skipped,
        "modalities_failed": failed,
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "model_size": args.model_size,
            "image_resolution": args.image_resolution,
            "batch_size": args.batch_size,
            "normalization": args.normalization,
            "max_patches_per_modality": args.max_patches_per_modality,
            "skip_existing": bool(args.skip_existing),
            "overwrite_embeddings": bool(args.overwrite_embeddings),
            "device_index": args.device_index,
            "force_cpu": bool(args.force_cpu),
        },
        "outputs": {key: path_to_str(value) for key, value in output_paths.items()},
        "modality_rows": modality_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract full CROMA embeddings for Instance C 224x224 patches."
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
        help="Default: <instance-root>/metadata/croma_probing/full_embeddings.",
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
        default=default_modalities(),
        help="Modalities to extract.",
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
        help="CROMA image resolution. Must equal patch size. Default: 224.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size. Use 1 for joint modalities if GPU memory is limited. Default: 2.",
    )

    parser.add_argument(
        "--normalization",
        choices=["per_sample_channel_clip", "per_batch_channel_clip", "none"],
        default="per_sample_channel_clip",
        help="Input normalization. Default: per_sample_channel_clip.",
    )

    parser.add_argument(
        "--max-patches-per-modality",
        type=int,
        default=0,
        help="Debug option. If >0, extract only first N patches per modality. Default: 0 = all.",
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
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip modality if output .npz already exists. Default: enabled.",
    )

    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Overwrite existing modality .npz embedding outputs.",
    )

    parser.add_argument(
        "--overwrite-summary",
        action="store_true",
        help="Overwrite summary CSV/JSON/Markdown outputs.",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Text progress interval if tqdm is unavailable. Default: 500.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    started_utc = now_utc()

    validate_requested_modalities(args.modalities)

    instance_root: Path = args.instance_root
    output_dir: Path = args.output_dir or (
        instance_root / "metadata" / "croma_probing" / "full_embeddings"
    )

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    comparison_manifest_csv: Path = args.comparison_manifest_csv or (
        instance_root / "metadata" / "croma_probing" / f"croma_comparison_manifest_{stem}.csv"
    )

    summary_csv = output_dir / f"croma_full_embedding_summary_{stem}.csv"
    json_path = output_dir / f"croma_full_embedding_summary_{stem}.json"
    md_path = output_dir / f"croma_full_embedding_summary_{stem}.md"

    output_paths = {
        "output_dir": output_dir,
        "summary_csv": summary_csv,
        "json": json_path,
        "markdown": md_path,
    }

    log("STEP", "Starting full CROMA embedding extraction.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Manifest CSV:  {path_to_str(comparison_manifest_csv)}")
    log("INFO", f"CROMA repo:    {path_to_str(args.croma_repo)}")
    log("INFO", f"Weights path:  {path_to_str(args.weights_path)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")
    log("INFO", f"Modalities:    {';'.join(args.modalities)}")
    log("INFO", f"Batch size:    {args.batch_size}")
    log("INFO", f"Normalization: {args.normalization}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not comparison_manifest_csv.exists():
        fail(f"Comparison manifest CSV does not exist: {path_to_str(comparison_manifest_csv)}")

    if not args.weights_path.exists():
        fail(f"CROMA weights path does not exist: {path_to_str(args.weights_path)}")

    if int(args.image_resolution) != int(args.patch_size):
        fail(
            f"image_resolution={args.image_resolution} must equal patch_size={args.patch_size}."
        )

    if int(args.image_resolution) % 8 != 0:
        fail("CROMA image_resolution must be divisible by 8.")

    output_dir.mkdir(parents=True, exist_ok=True)

    PretrainedCROMA = import_pretrained_croma(args.croma_repo)
    device = choose_device(int(args.device_index), bool(args.force_cpu))

    log("INFO", f"Selected device: {device}")

    manifest_rows = read_csv_rows(comparison_manifest_csv)
    validate_manifest_rows(manifest_rows)

    max_patches = int(args.max_patches_per_modality)
    max_patches_optional = max_patches if max_patches > 0 else None

    modality_results: List[Dict[str, object]] = []

    for modality in args.modalities:
        modality_rows = rows_for_modality(
            manifest_rows,
            modality=modality,
            max_patches=max_patches_optional,
        )

        result = extract_modality(
            modality=modality,
            rows=modality_rows,
            PretrainedCROMA=PretrainedCROMA,
            weights_path=args.weights_path,
            model_size=str(args.model_size),
            image_resolution=int(args.image_resolution),
            device=device,
            batch_size=int(args.batch_size),
            normalization=str(args.normalization),
            output_dir=output_dir,
            output_stem=stem,
            overwrite_embeddings=bool(args.overwrite_embeddings),
            skip_existing=bool(args.skip_existing),
            progress_every=int(args.progress_every),
        )

        modality_results.append(result)

        log(
            "OK" if result["status"] in {"completed", "skipped_existing"} else "ERROR",
            f"{modality}: status={result['status']}, "
            f"rows={result['n_rows']}, "
            f"shape={result['embedding_shape']}, "
            f"seconds={result['elapsed_seconds']}",
        )

        if result["status"] == "failed":
            log("ERROR", f"{modality} failed: {result['notes']}")

    summary = build_summary(
        instance_root=instance_root,
        comparison_manifest_csv=comparison_manifest_csv,
        croma_repo=args.croma_repo,
        weights_path=args.weights_path,
        device=device,
        modality_rows=modality_results,
        args=args,
        output_paths=output_paths,
        started_utc=started_utc,
    )

    log("STEP", "Writing extraction summary outputs.")

    write_csv(
        summary_csv,
        modality_results,
        overwrite=bool(args.overwrite_summary),
        fieldnames=[
            "modality",
            "status",
            "n_rows",
            "embedding_key",
            "embedding_shape",
            "output_npz",
            "croma_model_modality",
            "output_shapes",
            "elapsed_seconds",
            "notes",
        ],
    )

    write_json(json_path, summary, overwrite=bool(args.overwrite_summary))
    write_markdown(md_path, summary, modality_results, overwrite=bool(args.overwrite_summary))

    log("OK", f"Wrote summary CSV: {path_to_str(summary_csv)}")
    log("OK", f"Wrote JSON:        {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:    {path_to_str(md_path)}")

    log("STEP", "Final extraction summary.")
    log("OK" if summary["status"] == "passed" else "ERROR", f"Status: {summary['status']}")
    log("OK", f"Completed modalities: {summary['modalities_completed']}")
    log("OK", f"Skipped modalities: {summary['modalities_skipped']}")
    log("OK", f"Failed modalities: {summary['modalities_failed']}")

    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()