#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
4_extract_croma_dense_features_224.py

Main objective
--------------
Extract dense CROMA spatial features for the selected modality:

    s2_s1_snap_vv_vh

using the CROMA output key:

    joint_encodings

instead of the previous global key:

    joint_GAP

Why this script exists
----------------------
The previous global CROMA embeddings had shape:

    N x 768

Those are useful for frozen-probe classification, but not enough for UPerNet
segmentation.

For UPerNet we need dense spatial features, ideally saved as:

    N x 768 x 14 x 14

For 224 x 224 input patches, this corresponds to a 14 x 14 token grid when the
encoder uses 16 x 16 tokens.

Important
---------
Start with a smoke test:

python src/splitting_strategy_experiments/4_extract_croma_dense_features_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --croma-repo "C:/Users/acer/OneDrive/Desktop/UMR_espace_dev/CROMA" `
  --weights-path "D:/models/CROMA/CROMA_base.pt" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --target-modality "s2_s1_snap_vv_vh" `
  --feature-key "joint_encodings" `
  --model-size base `
  --image-resolution 224 `
  --batch-size 1 `
  --max-patches 20 `
  --overwrite

If the smoke test confirms a good dense shape, run the full extraction by
removing --max-patches.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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
# Constants
# ---------------------------------------------------------------------

EXPECTED_TARGET_MODALITY = "s2_s1_snap_vv_vh"
DEFAULT_FEATURE_KEY = "joint_encodings"


CITY_COL_CANDIDATES = [
    "city",
    "city_name",
    "municipality",
    "municipio",
]

REGION_COL_CANDIDATES = [
    "region",
    "brazil_region",
    "macroregion",
    "macro_region",
]

MODALITY_COL_CANDIDATES = [
    "modality",
    "input_modality",
    "data_modality",
]

PATCH_ID_COL_CANDIDATES = [
    "patch_id",
    "unique_patch_id",
    "patch_uid",
    "sample_id",
    "window_id",
]

MANIFEST_ROW_ID_COL_CANDIDATES = [
    "manifest_row_id",
    "manifest_row_ids",
    "row_id",
    "manifest_id",
]

S2_PATH_COL_CANDIDATES = [
    "s2_path",
    "s2_raster_path",
    "s2_file",
    "s2_filepath",
    "s2_final_path",
    "optical_path",
    "optical_raster_path",
    "sentinel2_path",
    "sentinel_2_path",
]

S1_PATH_COL_CANDIDATES = [
    "s1_path",
    "s1_raster_path",
    "s1_file",
    "s1_filepath",
    "s1_final_path",
    "s1_snap_path",
    "s1_snap_vv_vh_path",
    "sar_path",
    "sar_raster_path",
    "sentinel1_path",
    "sentinel_1_path",
]

ROW_OFF_COL_CANDIDATES = [
    "row_start",
    "row_off",
    "row_offset",
    "window_row_off",
    "window_row_offset",
    "y_start",
    "y_off",
    "y_offset",
    "patch_y",
    "top",
]

COL_OFF_COL_CANDIDATES = [
    "col_start",
    "col_off",
    "col_offset",
    "window_col_off",
    "window_col_offset",
    "x_start",
    "x_off",
    "x_offset",
    "patch_x",
    "left",
]

LABEL_BINARY_COL_CANDIDATES = [
    "label_binary",
    "is_positive",
    "positive",
    "has_favela",
    "contains_favela",
]

LABEL_POS_PIXELS_COL_CANDIDATES = [
    "label_positive_pixels",
    "favela_pixels",
    "positive_pixels",
    "n_positive_pixels",
    "label_sum",
    "mask_sum",
]

LABEL_POS_PERCENT_COL_CANDIDATES = [
    "label_positive_percent",
    "label_positive_percentage",
    "favela_percent",
    "favela_percentage",
    "favela_coverage",
    "positive_pixel_percentage",
]

LABEL_DENSITY_BIN_COL_CANDIDATES = [
    "label_density_bins",
    "label_density_bin",
    "density_bin",
    "favela_density_bin",
]


# ---------------------------------------------------------------------
# Logging and helpers
# ---------------------------------------------------------------------

def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    log("ERROR", message)
    raise SystemExit(exit_code)


def warn(message: str) -> None:
    log("WARNING", message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_to_str(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return str(path).replace("\\", "/")


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.size <= 20:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "preview": value.reshape(-1)[:20].tolist(),
        }
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if math.isnan(v):
            return None
        return v
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    return value


def clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_can_write(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            fail(
                "Output already exists and --overwrite was not used:\n"
                f"  {path_to_str(path)}"
            )


def find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    explicit: Optional[str] = None,
    required: bool = False,
    purpose: str = "column",
) -> Optional[str]:
    if explicit:
        if explicit in df.columns:
            return explicit
        fail(
            f"Requested {purpose} '{explicit}' was not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    norm_to_original = {normalize_text(c): c for c in df.columns}

    for candidate in candidates:
        norm = normalize_text(candidate)
        if norm in norm_to_original:
            return norm_to_original[norm]

    if required:
        fail(
            f"Could not detect {purpose}.\n"
            f"Tried candidates:\n{list(candidates)}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def default_manifest_path(instance_root: Path, patch_size: int, stride: int, edge_mode: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / f"croma_comparison_manifest_ps{patch_size}_st{stride}_{edge_mode}.csv"
    )


def default_output_dir(instance_root: Path, patch_size: int, stride: int, edge_mode: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "splitting_strategy_experiments"
        / f"croma_dense_features_ps{patch_size}_st{stride}_{edge_mode}"
    )


def output_stem(
    target_modality: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
    feature_key: str,
    max_patches: Optional[int],
    output_dtype: str,
) -> str:
    stem = (
        f"croma_dense_features_{target_modality}_"
        f"{feature_key}_ps{patch_size}_st{stride}_{edge_mode}_{output_dtype}"
    )
    if max_patches is not None:
        stem += f"_max{max_patches}"
    return stem


# ---------------------------------------------------------------------
# CROMA setup
# ---------------------------------------------------------------------

def import_pretrained_croma(croma_repo: Path):
    if not croma_repo.exists():
        fail(f"CROMA repo path does not exist:\n{path_to_str(croma_repo)}")

    use_croma_path = croma_repo / "use_croma.py"

    if not use_croma_path.exists():
        fail(f"use_croma.py not found:\n{path_to_str(use_croma_path)}")

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
        return torch.device(f"cuda:{device_index}")

    return torch.device("cpu")


def croma_model_modality_for_manifest_modality(modality: str) -> str:
    if modality == "s2":
        return "optical"

    if modality in {"s1_snap_vv_vh", "s1_rtc_vv_vh"}:
        return "SAR"

    if modality in {"s2_s1_snap_vv_vh", "s2_s1_rtc_vv_vh"}:
        return "both"

    fail(f"Unsupported modality: {modality}")


# ---------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------

def load_manifest(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    manifest_path = Path(args.comparison_manifest_csv) if args.comparison_manifest_csv else default_manifest_path(
        Path(args.instance_root),
        args.patch_size,
        args.stride,
        args.edge_mode,
    )

    if not manifest_path.exists():
        fail(f"Manifest CSV does not exist:\n{path_to_str(manifest_path)}")

    log("STEP", f"Reading manifest:\n{path_to_str(manifest_path)}")
    df = pd.read_csv(manifest_path)

    if df.empty:
        fail("Manifest is empty.")

    modality_col = find_column(
        df,
        MODALITY_COL_CANDIDATES,
        explicit=args.modality_col,
        required=True,
        purpose="modality column",
    )

    target_norm = normalize_text(args.target_modality)
    modality_norm = df[modality_col].map(normalize_text)
    df = df[modality_norm == target_norm].copy()

    if df.empty:
        fail(
            f"No rows found for target modality '{args.target_modality}' "
            f"using modality column '{modality_col}'."
        )

    patch_id_col = find_column(
        df,
        PATCH_ID_COL_CANDIDATES,
        explicit=args.patch_id_col,
        required=True,
        purpose="patch id column",
    )

    before = len(df)
    df = df.drop_duplicates(subset=[patch_id_col], keep="first").reset_index(drop=True)
    after = len(df)

    if before != after:
        warn(f"Deduplicated modality rows from {before:,} to {after:,} unique patches.")

    city_col = find_column(
        df,
        CITY_COL_CANDIDATES,
        explicit=args.city_col,
        required=True,
        purpose="city column",
    )

    region_col = find_column(
        df,
        REGION_COL_CANDIDATES,
        explicit=args.region_col,
        required=True,
        purpose="region column",
    )

    manifest_row_id_col = find_column(
        df,
        MANIFEST_ROW_ID_COL_CANDIDATES,
        explicit=args.manifest_row_id_col,
        required=False,
        purpose="manifest row id column",
    )

    s2_path_col = find_column(
        df,
        S2_PATH_COL_CANDIDATES,
        explicit=args.s2_path_col,
        required=True,
        purpose="Sentinel-2 raster path column",
    )

    s1_path_col = find_column(
        df,
        S1_PATH_COL_CANDIDATES,
        explicit=args.s1_path_col,
        required=True,
        purpose="Sentinel-1/SAR raster path column",
    )

    row_off_col = find_column(
        df,
        ROW_OFF_COL_CANDIDATES,
        explicit=args.row_off_col,
        required=True,
        purpose="window row offset column",
    )

    col_off_col = find_column(
        df,
        COL_OFF_COL_CANDIDATES,
        explicit=args.col_off_col,
        required=True,
        purpose="window column offset column",
    )

    label_binary_col = find_column(
        df,
        LABEL_BINARY_COL_CANDIDATES,
        explicit=args.label_binary_col,
        required=False,
        purpose="label binary column",
    )

    label_positive_pixels_col = find_column(
        df,
        LABEL_POS_PIXELS_COL_CANDIDATES,
        explicit=args.label_positive_pixels_col,
        required=False,
        purpose="label positive pixels column",
    )

    label_positive_percent_col = find_column(
        df,
        LABEL_POS_PERCENT_COL_CANDIDATES,
        explicit=args.label_positive_percent_col,
        required=False,
        purpose="label positive percent column",
    )

    label_density_bin_col = find_column(
        df,
        LABEL_DENSITY_BIN_COL_CANDIDATES,
        explicit=args.label_density_bin_col,
        required=False,
        purpose="label density bin column",
    )

    columns = {
        "manifest_path": str(manifest_path),
        "modality_col": modality_col,
        "patch_id_col": patch_id_col,
        "city_col": city_col,
        "region_col": region_col,
        "manifest_row_id_col": manifest_row_id_col,
        "s2_path_col": s2_path_col,
        "s1_path_col": s1_path_col,
        "row_off_col": row_off_col,
        "col_off_col": col_off_col,
        "label_binary_col": label_binary_col,
        "label_positive_pixels_col": label_positive_pixels_col,
        "label_positive_percent_col": label_positive_percent_col,
        "label_density_bin_col": label_density_bin_col,
    }

    log("INFO", "Detected columns:")
    for key, value in columns.items():
        if key != "manifest_path":
            log("INFO", f"  {key}: {value}")

    return df, columns


def sample_manifest_rows(
    df: pd.DataFrame,
    args: argparse.Namespace,
    columns: Dict[str, Optional[str]],
) -> pd.DataFrame:
    if args.max_patches is None:
        return df.reset_index(drop=True)

    max_patches = int(args.max_patches)

    if max_patches <= 0:
        fail("--max-patches must be positive if provided.")

    if len(df) <= max_patches:
        return df.reset_index(drop=True)

    if args.sample_mode == "first":
        return df.head(max_patches).reset_index(drop=True)

    if args.sample_mode == "random":
        return df.sample(n=max_patches, random_state=args.random_seed).reset_index(drop=True)

    # Balanced mode: use positive/empty if possible.
    label_positive_pixels_col = columns.get("label_positive_pixels_col")
    label_binary_col = columns.get("label_binary_col")

    if label_positive_pixels_col is not None:
        positive_mask = pd.to_numeric(df[label_positive_pixels_col], errors="coerce").fillna(0) > 0
    elif label_binary_col is not None:
        positive_mask = pd.to_numeric(df[label_binary_col], errors="coerce").fillna(0) > 0
    else:
        warn("No label column found for balanced sampling. Falling back to first rows.")
        return df.head(max_patches).reset_index(drop=True)

    pos_df = df[positive_mask].copy()
    neg_df = df[~positive_mask].copy()

    n_pos = min(len(pos_df), max_patches // 2)
    n_neg = min(len(neg_df), max_patches - n_pos)

    selected_parts = []

    if n_pos > 0:
        selected_parts.append(pos_df.sample(n=n_pos, random_state=args.random_seed))

    if n_neg > 0:
        selected_parts.append(neg_df.sample(n=n_neg, random_state=args.random_seed))

    selected = pd.concat(selected_parts, axis=0)

    if len(selected) < max_patches:
        remaining = df.drop(index=selected.index)
        extra_n = min(max_patches - len(selected), len(remaining))
        if extra_n > 0:
            selected = pd.concat(
                [
                    selected,
                    remaining.sample(n=extra_n, random_state=args.random_seed),
                ],
                axis=0,
            )

    selected = selected.sample(frac=1.0, random_state=args.random_seed).reset_index(drop=True)
    return selected


# ---------------------------------------------------------------------
# Raster loading and normalization
# ---------------------------------------------------------------------

def resolve_path(path_value: Any, instance_root: Path) -> Path:
    raw = str(path_value).strip().replace("\\", "/")

    if raw == "":
        fail("Encountered empty raster path in manifest.")

    path = Path(raw)

    if path.exists():
        return path

    if not path.is_absolute():
        candidate = instance_root / raw
        if candidate.exists():
            return candidate

    fail(
        "Raster path from manifest does not exist:\n"
        f"  original: {raw}\n"
        f"  tried as: {path_to_str(path)}"
    )


def read_raster_window(
    raster_path: Path,
    row_off: int,
    col_off: int,
    patch_size: int,
    band_indexes: Sequence[int],
) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        available_bands = src.count

        for band in band_indexes:
            if band < 1 or band > available_bands:
                fail(
                    f"Requested band {band} from raster with {available_bands} bands:\n"
                    f"{path_to_str(raster_path)}"
                )

        window = Window(
            col_off=int(col_off),
            row_off=int(row_off),
            width=int(patch_size),
            height=int(patch_size),
        )

        arr = src.read(
            indexes=list(band_indexes),
            window=window,
            boundless=True,
            fill_value=np.nan,
            out_shape=(len(band_indexes), patch_size, patch_size),
        ).astype(np.float32)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def normalize_like_croma_readme(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-sample, per-channel normalization.

    For each sample and channel:
      1. compute mean/std over H,W
      2. clip to mean ± 2 std
      3. scale to [0,1]

    This follows the style used in the previous CROMA extraction scripts.
    """
    x = x.float()

    b, c, h, w = x.shape
    flat = x.view(b, c, -1)

    mean = flat.mean(dim=2, keepdim=True)
    std = flat.std(dim=2, keepdim=True).clamp_min(eps)

    low = mean - 2.0 * std
    high = mean + 2.0 * std

    flat = torch.maximum(torch.minimum(flat, high), low)
    flat = (flat - low) / (high - low + eps)

    return flat.view(b, c, h, w)


def build_batch(
    batch_df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
    instance_root: Path,
    patch_size: int,
    device: torch.device,
    normalize_inputs: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s2_arrays: List[np.ndarray] = []
    sar_arrays: List[np.ndarray] = []

    s2_path_col = columns["s2_path_col"]
    s1_path_col = columns["s1_path_col"]
    row_off_col = columns["row_off_col"]
    col_off_col = columns["col_off_col"]

    assert s2_path_col is not None
    assert s1_path_col is not None
    assert row_off_col is not None
    assert col_off_col is not None

    for _, row in batch_df.iterrows():
        row_off = safe_int(row[row_off_col])
        col_off = safe_int(row[col_off_col])

        s2_path = resolve_path(row[s2_path_col], instance_root)
        s1_path = resolve_path(row[s1_path_col], instance_root)

        # Sentinel-2: use 12 optical bands.
        s2 = read_raster_window(
            raster_path=s2_path,
            row_off=row_off,
            col_off=col_off,
            patch_size=patch_size,
            band_indexes=list(range(1, 13)),
        )

        # Sentinel-1 SNAP-GRD: use VV/VH only.
        # If the raster has VV, VH, VV_minus_VH, this intentionally ignores band 3.
        sar = read_raster_window(
            raster_path=s1_path,
            row_off=row_off,
            col_off=col_off,
            patch_size=patch_size,
            band_indexes=[1, 2],
        )

        s2_arrays.append(s2)
        sar_arrays.append(sar)

    s2_np = np.stack(s2_arrays, axis=0).astype(np.float32)
    sar_np = np.stack(sar_arrays, axis=0).astype(np.float32)

    optical_tensor = torch.from_numpy(s2_np).to(device)
    sar_tensor = torch.from_numpy(sar_np).to(device)

    if normalize_inputs:
        optical_tensor = normalize_like_croma_readme(optical_tensor)
        sar_tensor = normalize_like_croma_readme(sar_tensor)

    return sar_tensor, optical_tensor


# ---------------------------------------------------------------------
# Dense feature conversion
# ---------------------------------------------------------------------

def square_hw_from_tokens(tokens: int) -> Tuple[int, int]:
    hw = int(round(math.sqrt(tokens)))
    if hw * hw != tokens:
        fail(f"Token count {tokens} is not a perfect square, cannot reshape to HxW.")
    return hw, hw


def convert_croma_output_to_nchw(
    tensor: torch.Tensor,
    *,
    feature_key: str,
    remove_cls_token: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert CROMA dense output into N x C x H x W numpy array.

    Supported input forms:
      B x tokens x C
      B x C x tokens
      B x C x H x W
      B x H x W x C
    """
    raw_shape = tuple(int(x) for x in tensor.shape)

    x = tensor.detach().float().cpu()

    info: Dict[str, Any] = {
        "feature_key": feature_key,
        "raw_shape": raw_shape,
        "converted_shape": None,
        "conversion": None,
        "removed_cls_token": False,
    }

    if x.ndim == 4:
        b, d1, d2, d3 = x.shape

        # Already B x C x H x W
        if d1 in {768, 384, 512, 1024} and d2 > 1 and d3 > 1:
            arr = x.numpy().astype(np.float32)
            info["converted_shape"] = tuple(arr.shape)
            info["conversion"] = "already_b_c_h_w"
            return arr, info

        # B x H x W x C
        if d3 in {768, 384, 512, 1024} and d1 > 1 and d2 > 1:
            arr = x.permute(0, 3, 1, 2).contiguous().numpy().astype(np.float32)
            info["converted_shape"] = tuple(arr.shape)
            info["conversion"] = "b_h_w_c_to_b_c_h_w"
            return arr, info

        fail(
            f"Unsupported 4D dense feature shape for key '{feature_key}': {raw_shape}"
        )

    if x.ndim == 3:
        b, a, c = x.shape

        # B x tokens x C
        if c in {768, 384, 512, 1024}:
            tokens = a

            if remove_cls_token and tokens > 1:
                possible_tokens = tokens - 1
                possible_hw = int(round(math.sqrt(possible_tokens)))
                if possible_hw * possible_hw == possible_tokens:
                    x = x[:, 1:, :]
                    tokens = possible_tokens
                    info["removed_cls_token"] = True

            h, w = square_hw_from_tokens(tokens)

            # B x tokens x C -> B x C x H x W
            arr = (
                x.reshape(b, h, w, c)
                .permute(0, 3, 1, 2)
                .contiguous()
                .numpy()
                .astype(np.float32)
            )

            info["converted_shape"] = tuple(arr.shape)
            info["conversion"] = "b_tokens_c_to_b_c_h_w"
            return arr, info

        # B x C x tokens
        if a in {768, 384, 512, 1024}:
            channels = a
            tokens = c

            if remove_cls_token and tokens > 1:
                possible_tokens = tokens - 1
                possible_hw = int(round(math.sqrt(possible_tokens)))
                if possible_hw * possible_hw == possible_tokens:
                    x = x[:, :, 1:]
                    tokens = possible_tokens
                    info["removed_cls_token"] = True

            h, w = square_hw_from_tokens(tokens)

            arr = (
                x.reshape(b, channels, h, w)
                .contiguous()
                .numpy()
                .astype(np.float32)
            )

            info["converted_shape"] = tuple(arr.shape)
            info["conversion"] = "b_c_tokens_to_b_c_h_w"
            return arr, info

        fail(
            f"Unsupported 3D dense feature shape for key '{feature_key}': {raw_shape}"
        )

    fail(
        f"Feature key '{feature_key}' produced unsupported ndim={x.ndim}, shape={raw_shape}."
    )


# ---------------------------------------------------------------------
# Metadata output
# ---------------------------------------------------------------------

def metadata_array(
    df: pd.DataFrame,
    col: Optional[str],
    default: Any,
    dtype: Any,
) -> np.ndarray:
    if col is None:
        values = [default] * len(df)
    else:
        values = df[col].tolist()

    return np.asarray(values, dtype=dtype)


def save_metadata_npz(
    path: Path,
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
    args: argparse.Namespace,
    feature_path: Path,
    feature_shape: Tuple[int, int, int, int],
    raw_feature_shapes: List[Dict[str, Any]],
    overwrite: bool,
) -> None:
    ensure_can_write(path, overwrite)

    manifest_row_id_col = columns.get("manifest_row_id_col")
    patch_id_col = columns.get("patch_id_col")
    city_col = columns.get("city_col")
    region_col = columns.get("region_col")
    label_binary_col = columns.get("label_binary_col")
    label_positive_pixels_col = columns.get("label_positive_pixels_col")
    label_positive_percent_col = columns.get("label_positive_percent_col")
    label_density_bin_col = columns.get("label_density_bin_col")

    if manifest_row_id_col is None:
        manifest_row_ids = np.asarray([str(i) for i in range(len(df))], dtype=str)
    else:
        manifest_row_ids = metadata_array(df, manifest_row_id_col, "", str)

    patch_ids = metadata_array(df, patch_id_col, "", str)
    cities = metadata_array(df, city_col, "", str)
    regions = metadata_array(df, region_col, "", str)

    if label_binary_col is not None:
        label_binary = pd.to_numeric(df[label_binary_col], errors="coerce").fillna(0).astype(np.int64).values
    elif label_positive_pixels_col is not None:
        label_binary = (
            pd.to_numeric(df[label_positive_pixels_col], errors="coerce")
            .fillna(0)
            .gt(0)
            .astype(np.int64)
            .values
        )
    else:
        label_binary = np.zeros(len(df), dtype=np.int64)

    if label_positive_pixels_col is not None:
        label_positive_pixels = (
            pd.to_numeric(df[label_positive_pixels_col], errors="coerce")
            .fillna(0)
            .astype(np.int64)
            .values
        )
    else:
        label_positive_pixels = np.zeros(len(df), dtype=np.int64)

    if label_positive_percent_col is not None:
        label_positive_percent = (
            pd.to_numeric(df[label_positive_percent_col], errors="coerce")
            .fillna(0)
            .astype(np.float32)
            .values
        )
    else:
        label_positive_percent = np.zeros(len(df), dtype=np.float32)

    label_density_bins = metadata_array(df, label_density_bin_col, "", str)

    np.savez_compressed(
        path,
        manifest_row_ids=manifest_row_ids,
        patch_ids=patch_ids,
        cities=cities,
        regions=regions,
        label_binary=label_binary,
        label_positive_pixels=label_positive_pixels,
        label_positive_percent=label_positive_percent.astype(np.float32),
        label_density_bins=label_density_bins,
        target_modality=np.asarray([args.target_modality], dtype=str),
        croma_model_modality=np.asarray(["both"], dtype=str),
        feature_key=np.asarray([args.feature_key], dtype=str),
        dense_feature_path=np.asarray([path_to_str(feature_path)], dtype=str),
        dense_feature_shape=np.asarray(feature_shape, dtype=np.int64),
        raw_feature_shapes_json=np.asarray([json.dumps(jsonable(raw_feature_shapes))], dtype=str),
        model_size=np.asarray([args.model_size], dtype=str),
        image_resolution=np.asarray([args.image_resolution], dtype=np.int64),
        patch_size=np.asarray([args.patch_size], dtype=np.int64),
        stride=np.asarray([args.stride], dtype=np.int64),
        edge_mode=np.asarray([args.edge_mode], dtype=str),
        output_dtype=np.asarray([args.output_dtype], dtype=str),
        normalization=np.asarray([args.normalization], dtype=str),
        created_utc=np.asarray([now_utc()], dtype=str),
    )


def write_manifest_csv(
    path: Path,
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]],
    overwrite: bool,
) -> None:
    ensure_can_write(path, overwrite)

    out = pd.DataFrame()
    out["dense_index"] = np.arange(len(df), dtype=np.int64)

    for out_col, source_key in [
        ("manifest_row_id", "manifest_row_id_col"),
        ("patch_id", "patch_id_col"),
        ("city", "city_col"),
        ("region", "region_col"),
        ("label_binary", "label_binary_col"),
        ("label_positive_pixels", "label_positive_pixels_col"),
        ("label_positive_percent", "label_positive_percent_col"),
        ("label_density_bin", "label_density_bin_col"),
        ("s2_path", "s2_path_col"),
        ("s1_path", "s1_path_col"),
        ("row_off", "row_off_col"),
        ("col_off", "col_off_col"),
    ]:
        source_col = columns.get(source_key)
        if source_col is not None:
            out[out_col] = df[source_col].values
        else:
            out[out_col] = ""

    out.to_csv(path, index=False)


def write_json(path: Path, payload: Dict[str, Any], overwrite: bool) -> None:
    ensure_can_write(path, overwrite)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)


def write_markdown_report(
    path: Path,
    summary: Dict[str, Any],
    overwrite: bool,
) -> None:
    ensure_can_write(path, overwrite)

    lines: List[str] = []

    lines.append("# Dense CROMA Feature Extraction Report")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Extract dense CROMA spatial features for the S2 + SNAP-GRD VV/VH modality "
        "using `joint_encodings`, so that a UPerNet-style segmentation decoder can be trained."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for key in [
        "instance_root",
        "manifest_path",
        "croma_repo",
        "weights_path",
        "target_modality",
        "feature_key",
        "model_size",
        "image_resolution",
        "patch_size",
        "stride",
        "edge_mode",
        "batch_size",
        "max_patches",
        "output_dtype",
        "normalization",
        "device",
    ]:
        lines.append(f"- {key}: `{summary.get(key)}`")

    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Dense feature file: `{summary['dense_feature_path']}`")
    lines.append(f"- Metadata file: `{summary['metadata_npz_path']}`")
    lines.append(f"- Dense manifest CSV: `{summary['dense_manifest_csv_path']}`")
    lines.append(f"- Summary JSON: `{summary['summary_json_path']}`")
    lines.append("")
    lines.append("## Shape")
    lines.append("")
    lines.append(f"- Final dense feature shape: `{summary['dense_feature_shape']}`")
    lines.append(f"- First raw CROMA feature shape: `{summary['first_raw_feature_shape']}`")
    lines.append(f"- First conversion: `{summary['first_conversion']}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The final dense feature tensor is saved in `N x C x H x W` format. "
        "For 224x224 patches, a typical expected shape is `N x 768 x 14 x 14`."
    )
    lines.append("")
    lines.append("## Batch diagnostics")
    lines.append("")
    lines.append("| Batch | Start | End | Batch size | Raw shape | Converted shape | Seconds |")
    lines.append("|---:|---:|---:|---:|---|---|---:|")

    for row in summary.get("batch_rows", []):
        lines.append(
            f"| {row['batch_index']} "
            f"| {row['start']} "
            f"| {row['end']} "
            f"| {row['batch_size']} "
            f"| `{row['raw_shape']}` "
            f"| `{row['converted_shape']}` "
            f"| {row['seconds']} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------

def extract_dense_features(args: argparse.Namespace) -> None:
    started = time.time()

    instance_root = Path(args.instance_root)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
        instance_root,
        args.patch_size,
        args.stride,
        args.edge_mode,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = output_stem(
        target_modality=args.target_modality,
        patch_size=args.patch_size,
        stride=args.stride,
        edge_mode=args.edge_mode,
        feature_key=args.feature_key,
        max_patches=args.max_patches,
        output_dtype=args.output_dtype,
    )

    dense_feature_path = output_dir / f"{stem}.npy"
    metadata_npz_path = output_dir / f"{stem}_metadata.npz"
    dense_manifest_csv_path = output_dir / f"{stem}_manifest.csv"
    batch_csv_path = output_dir / f"{stem}_batch_log.csv"
    summary_json_path = output_dir / f"{stem}_summary.json"
    report_md_path = output_dir / f"{stem}_report.md"

    log("=" * 100, "")
    log("STEP", "Dense CROMA feature extraction")
    log("INFO", f"Instance root:   {path_to_str(instance_root)}")
    log("INFO", f"CROMA repo:      {path_to_str(Path(args.croma_repo))}")
    log("INFO", f"Weights path:    {path_to_str(Path(args.weights_path))}")
    log("INFO", f"Output dir:      {path_to_str(output_dir)}")
    log("INFO", f"Target modality: {args.target_modality}")
    log("INFO", f"Feature key:     {args.feature_key}")
    log("INFO", f"Max patches:     {args.max_patches}")
    log("=" * 100, "")

    if args.target_modality != EXPECTED_TARGET_MODALITY:
        warn(
            f"This script was designed for {EXPECTED_TARGET_MODALITY}, "
            f"but got {args.target_modality}."
        )

    if args.feature_key == "joint_GAP":
        fail(
            "You selected joint_GAP, which is a global embedding. "
            "For dense extraction use --feature-key joint_encodings."
        )

    if int(args.patch_size) != int(args.image_resolution):
        fail("--patch-size must match --image-resolution for CROMA.")

    if args.output_dtype not in {"float16", "float32"}:
        fail("--output-dtype must be float16 or float32.")

    output_np_dtype = np.float16 if args.output_dtype == "float16" else np.float32

    df, columns = load_manifest(args)
    df = sample_manifest_rows(df, args, columns)

    n_total = len(df)
    if n_total == 0:
        fail("No rows selected for extraction.")

    log("INFO", f"Selected patches for extraction: {n_total:,}")

    PretrainedCROMA = import_pretrained_croma(Path(args.croma_repo))
    device = choose_device(int(args.device_index), bool(args.force_cpu))
    log("INFO", f"Selected device: {device}")

    model_modality = croma_model_modality_for_manifest_modality(args.target_modality)
    if model_modality != "both":
        fail(
            f"Expected model modality 'both' for {args.target_modality}, got {model_modality}."
        )

    if not Path(args.weights_path).exists():
        fail(f"CROMA weights path does not exist:\n{path_to_str(Path(args.weights_path))}")

    model = None
    feature_memmap = None
    batch_rows: List[Dict[str, Any]] = []
    raw_feature_shapes: List[Dict[str, Any]] = []

    try:
        log("STEP", "Loading CROMA model")
        model = PretrainedCROMA(
            pretrained_path=str(args.weights_path),
            size=str(args.model_size),
            modality=model_modality,
            image_resolution=int(args.image_resolution),
        ).to(device)

        model.eval()

        ensure_can_write(dense_feature_path, args.overwrite)

        progress = None
        if tqdm is not None:
            progress = tqdm(total=n_total, desc="Dense CROMA", unit="patch")

        first_batch_done = False
        write_index = 0

        with torch.no_grad():
            for batch_index, start in enumerate(range(0, n_total, int(args.batch_size))):
                batch_started = time.time()

                end = min(start + int(args.batch_size), n_total)
                batch_df = df.iloc[start:end].copy()

                sar_tensor, optical_tensor = build_batch(
                    batch_df=batch_df,
                    columns=columns,
                    instance_root=instance_root,
                    patch_size=int(args.patch_size),
                    device=device,
                    normalize_inputs=(args.normalization == "per_sample_channel"),
                )

                outputs = model(
                    SAR_images=sar_tensor,
                    optical_images=optical_tensor,
                )

                if not isinstance(outputs, dict):
                    fail(
                        "CROMA output is not a dictionary. "
                        f"Got type: {type(outputs)}"
                    )

                if args.feature_key not in outputs:
                    shapes = {
                        key: tuple(value.shape) if hasattr(value, "shape") else str(type(value))
                        for key, value in outputs.items()
                    }
                    fail(
                        f"Feature key '{args.feature_key}' not found in CROMA outputs.\n"
                        f"Available keys and shapes:\n{json.dumps(jsonable(shapes), indent=2)}"
                    )

                raw_tensor = outputs[args.feature_key]
                dense_np, conversion_info = convert_croma_output_to_nchw(
                    raw_tensor,
                    feature_key=args.feature_key,
                    remove_cls_token=bool(args.remove_cls_token),
                )

                if dense_np.shape[0] != len(batch_df):
                    fail(
                        f"Batch output mismatch: dense batch has {dense_np.shape[0]} rows, "
                        f"but input batch has {len(batch_df)} rows."
                    )

                if not first_batch_done:
                    feature_shape = (
                        int(n_total),
                        int(dense_np.shape[1]),
                        int(dense_np.shape[2]),
                        int(dense_np.shape[3]),
                    )

                    log("INFO", f"First raw CROMA shape: {conversion_info['raw_shape']}")
                    log("INFO", f"Converted dense shape for first batch: {tuple(dense_np.shape)}")
                    log("INFO", f"Final dense feature file shape: {feature_shape}")
                    log("INFO", f"Conversion: {conversion_info['conversion']}")

                    feature_memmap = np.lib.format.open_memmap(
                        dense_feature_path,
                        mode="w+",
                        dtype=output_np_dtype,
                        shape=feature_shape,
                    )

                    first_batch_done = True

                assert feature_memmap is not None

                batch_n = dense_np.shape[0]
                feature_memmap[write_index:write_index + batch_n] = dense_np.astype(output_np_dtype)
                feature_memmap.flush()

                raw_feature_shapes.append(conversion_info)

                batch_seconds = round(time.time() - batch_started, 3)
                batch_rows.append(
                    {
                        "batch_index": batch_index,
                        "start": start,
                        "end": end,
                        "batch_size": batch_n,
                        "raw_shape": str(conversion_info["raw_shape"]),
                        "converted_shape": str(tuple(dense_np.shape)),
                        "conversion": conversion_info["conversion"],
                        "removed_cls_token": conversion_info["removed_cls_token"],
                        "seconds": batch_seconds,
                    }
                )

                write_index += batch_n

                if progress is not None:
                    progress.update(batch_n)

                if int(args.progress_every) > 0 and batch_index % int(args.progress_every) == 0:
                    log(
                        "INFO",
                        f"Batch {batch_index}: wrote {write_index:,}/{n_total:,} patches, "
                        f"raw={conversion_info['raw_shape']}, dense={tuple(dense_np.shape)}"
                    )

                del sar_tensor, optical_tensor, outputs, raw_tensor, dense_np
                clear_torch_memory()

        if progress is not None:
            progress.close()

        if feature_memmap is None:
            fail("No features were written.")

        final_feature_shape = tuple(int(x) for x in feature_memmap.shape)

        if final_feature_shape[0] != n_total:
            fail(
                f"Final feature row mismatch: {final_feature_shape[0]} != {n_total}"
            )

        # Save companion metadata and manifests.
        save_metadata_npz(
            path=metadata_npz_path,
            df=df,
            columns=columns,
            args=args,
            feature_path=dense_feature_path,
            feature_shape=final_feature_shape,
            raw_feature_shapes=raw_feature_shapes,
            overwrite=args.overwrite,
        )

        write_manifest_csv(
            path=dense_manifest_csv_path,
            df=df,
            columns=columns,
            overwrite=args.overwrite,
        )

        pd.DataFrame(batch_rows).to_csv(batch_csv_path, index=False)

        elapsed = round(time.time() - started, 3)

        summary = {
            "status": "completed",
            "instance_root": path_to_str(instance_root),
            "manifest_path": columns["manifest_path"],
            "croma_repo": path_to_str(Path(args.croma_repo)),
            "weights_path": path_to_str(Path(args.weights_path)),
            "target_modality": args.target_modality,
            "feature_key": args.feature_key,
            "model_size": args.model_size,
            "image_resolution": args.image_resolution,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "batch_size": args.batch_size,
            "max_patches": args.max_patches,
            "output_dtype": args.output_dtype,
            "normalization": args.normalization,
            "device": str(device),
            "n_patches": n_total,
            "dense_feature_shape": str(final_feature_shape),
            "first_raw_feature_shape": str(raw_feature_shapes[0]["raw_shape"]) if raw_feature_shapes else "",
            "first_conversion": raw_feature_shapes[0]["conversion"] if raw_feature_shapes else "",
            "dense_feature_path": path_to_str(dense_feature_path),
            "metadata_npz_path": path_to_str(metadata_npz_path),
            "dense_manifest_csv_path": path_to_str(dense_manifest_csv_path),
            "batch_csv_path": path_to_str(batch_csv_path),
            "summary_json_path": path_to_str(summary_json_path),
            "report_md_path": path_to_str(report_md_path),
            "columns": columns,
            "batch_rows": batch_rows,
            "raw_feature_shapes_first_10": raw_feature_shapes[:10],
            "elapsed_seconds": elapsed,
            "created_utc": now_utc(),
        }

        write_json(summary_json_path, summary, overwrite=args.overwrite)
        write_markdown_report(report_md_path, summary, overwrite=args.overwrite)

        log("OK", "Dense CROMA feature extraction completed.")
        log("OK", f"Dense feature file: {path_to_str(dense_feature_path)}")
        log("OK", f"Metadata NPZ:       {path_to_str(metadata_npz_path)}")
        log("OK", f"Manifest CSV:       {path_to_str(dense_manifest_csv_path)}")
        log("OK", f"Summary JSON:       {path_to_str(summary_json_path)}")
        log("OK", f"Report MD:          {path_to_str(report_md_path)}")
        log("OK", f"Final feature shape: {final_feature_shape}")

    except Exception:
        traceback.print_exc()
        fail("Dense feature extraction failed. See traceback above.")

    finally:
        if model is not None:
            del model
        if feature_memmap is not None:
            del feature_memmap
        clear_torch_memory()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract dense CROMA features for UPerNet LORO segmentation."
    )

    parser.add_argument(
        "--instance-root",
        required=True,
        help="Dataset instance root.",
    )
    parser.add_argument(
        "--comparison-manifest-csv",
        default=None,
        help="Optional explicit manifest CSV path.",
    )
    parser.add_argument(
        "--croma-repo",
        required=True,
        help="Path to official CROMA repo containing use_croma.py.",
    )
    parser.add_argument(
        "--weights-path",
        required=True,
        help="Path to CROMA_base.pt or CROMA_large.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default is metadata/splitting_strategy_experiments/croma_dense_features_*.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=112,
        help="Patch stride.",
    )
    parser.add_argument(
        "--edge-mode",
        default="cover",
        help="Edge mode.",
    )
    parser.add_argument(
        "--target-modality",
        default=EXPECTED_TARGET_MODALITY,
        help="Target modality. Default: s2_s1_snap_vv_vh.",
    )
    parser.add_argument(
        "--feature-key",
        default=DEFAULT_FEATURE_KEY,
        help="Dense CROMA output key. Default: joint_encodings.",
    )
    parser.add_argument(
        "--model-size",
        choices=["base", "large"],
        default="base",
        help="CROMA model size.",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=224,
        help="CROMA image resolution. Must equal patch size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size. Start with 1 for joint dense features.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Optional smoke-test limit. Example: --max-patches 20.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random", "balanced"],
        default="balanced",
        help="How to select rows when --max-patches is used.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for random/balanced smoke-test sampling.",
    )

    parser.add_argument(
        "--normalization",
        choices=["per_sample_channel", "none"],
        default="per_sample_channel",
        help="Input normalization strategy.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Dense feature storage dtype. float16 saves disk space.",
    )
    parser.add_argument(
        "--remove-cls-token",
        action="store_true",
        default=True,
        help="Remove CLS token if dense output has 197 tokens. Default: enabled.",
    )
    parser.add_argument(
        "--keep-cls-token",
        dest="remove_cls_token",
        action="store_false",
        help="Do not remove CLS token. Usually not recommended for segmentation.",
    )

    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU. Not recommended for full extraction.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N batches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    # Optional explicit column names.
    parser.add_argument("--modality-col", default=None)
    parser.add_argument("--patch-id-col", default=None)
    parser.add_argument("--manifest-row-id-col", default=None)
    parser.add_argument("--city-col", default=None)
    parser.add_argument("--region-col", default=None)
    parser.add_argument("--s2-path-col", default=None)
    parser.add_argument("--s1-path-col", default=None)
    parser.add_argument("--row-off-col", default=None)
    parser.add_argument("--col-off-col", default=None)
    parser.add_argument("--label-binary-col", default=None)
    parser.add_argument("--label-positive-pixels-col", default=None)
    parser.add_argument("--label-positive-percent-col", default=None)
    parser.add_argument("--label-density-bin-col", default=None)

    return parser.parse_args()


if __name__ == "__main__":
    extract_dense_features(parse_args())