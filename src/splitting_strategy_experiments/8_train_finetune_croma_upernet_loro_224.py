#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
8_train_finetune_croma_upernet_loro_224.py

Main objective
--------------
Fine-tune CROMA + a UPerNet-style segmentation decoder end-to-end for
Leave-One-Region-Out favela segmentation.

This script does NOT use precomputed dense CROMA features.
Instead, it reads raw S2 + SNAP-GRD VV/VH patches, runs PretrainedCROMA,
takes `joint_encodings`, converts them to B x C x H x W, and trains the
segmentation decoder.

Why this script exists
----------------------
The frozen dense-feature baseline learned reasonable validation behaviour,
but Southeast test generalization remained weak. This script tests whether
adapting the CROMA encoder itself improves regional generalization.

Recommended smoke test
----------------------
python src/splitting_strategy_experiments/8_train_finetune_croma_upernet_loro_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --croma-repo "C:/Users/acer/OneDrive/Desktop/UMR_espace_dev/CROMA" `
  --weights-path "D:/models/CROMA/CROMA_base.pt" `
  --heldout-region Southeast `
  --epochs 2 `
  --batch-size 1 `
  --grad-accum-steps 8 `
  --max-train-batches 8 `
  --max-val-batches 4 `
  --max-test-batches 4 `
  --num-workers 0 `
  --freeze-croma-mode none `
  --max-pos-weight 10 `
  --run-name "smoke_finetune_croma_upernet_Southeast_bs1_acc8" `
  --overwrite

Recommended first full pilot
----------------------------
python src/splitting_strategy_experiments/8_train_finetune_croma_upernet_loro_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --croma-repo "C:/Users/acer/OneDrive/Desktop/UMR_espace_dev/CROMA" `
  --weights-path "D:/models/CROMA/CROMA_base.pt" `
  --heldout-region Southeast `
  --epochs 20 `
  --batch-size 1 `
  --grad-accum-steps 16 `
  --num-workers 0 `
  --freeze-croma-mode none `
  --encoder-lr 1e-5 `
  --decoder-lr 3e-4 `
  --max-pos-weight 10 `
  --run-name "heldout_Southeast_finetune_croma_upernet_epochs20_bs1_acc16_posw10" `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
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
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "[ERROR] PyTorch is required.\n"
        "Install PyTorch first.\n\n"
        f"Original error: {exc}"
    )

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


REGIONS = ["Central-West", "North", "Northeast", "South", "Southeast"]


# ---------------------------------------------------------------------
# Logging and utility helpers
# ---------------------------------------------------------------------

def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def warn(message: str) -> None:
    log("WARNING", message)


def fail(message: str, exit_code: int = 1) -> None:
    log("ERROR", message)
    raise SystemExit(exit_code)


def banner(title: str) -> None:
    print("[" + "=" * 100 + "]", flush=True)
    log("STEP", title)
    print("[" + "=" * 100 + "]", flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_to_str(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return str(path).replace("\\", "/")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return path_to_str(value)
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
        return None if math.isnan(v) else v
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def choose_device(force_cpu: bool, device_index: int) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_run_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        fail(
            "Run directory already exists and is not empty:\n"
            f"{path_to_str(path)}\n\n"
            "Use --overwrite to reuse it."
        )
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def slug_region(region: str) -> str:
    text = str(region).strip()
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text


def safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def parse_band_indices(value: Any, default: Sequence[int]) -> List[int]:
    """
    Manifest columns may store band indices as strings such as:
      "1,2,3"
      "[1, 2, 3]"
      "1;2;3"

    This returns 1-based raster band indices.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return list(default)

    text = str(value).strip()
    if not text:
        return list(default)

    text = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    parts = re.split(r"[,\s;]+", text)
    out: List[int] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except Exception:
            pass

    return out if out else list(default)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def default_alignment_dir(instance_root: Path) -> Path:
    return (
        instance_root
        / "metadata"
        / "splitting_strategy_experiments"
        / "dense_loro_alignment_validation_ps224_st112_cover"
    )


def split_csv_path(alignment_dir: Path, heldout_region: str, split: str) -> Path:
    return alignment_dir / f"loro_fold_{slug_region(heldout_region)}_{split}_with_dense_index.csv"


def default_training_root(instance_root: Path) -> Path:
    return (
        instance_root
        / "experiments"
        / "splitting_strategy_experiments"
        / "finetune_croma_upernet_loro_ps224_st112_cover"
    )


# ---------------------------------------------------------------------
# CROMA loading
# ---------------------------------------------------------------------

def import_pretrained_croma(croma_repo: Path):
    if not croma_repo.exists():
        fail(f"CROMA repository path does not exist:\n{path_to_str(croma_repo)}")

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


def count_parameters(module: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(frozen),
    }


def apply_croma_freezing(
    croma: nn.Module,
    freeze_mode: str,
    trainable_substrings: Sequence[str],
) -> Dict[str, Any]:
    """
    freeze_mode:
      - none: all CROMA parameters trainable
      - all: all CROMA parameters frozen
      - substrings: freeze all, then unfreeze params whose name contains one of trainable_substrings
    """
    if freeze_mode == "none":
        for p in croma.parameters():
            p.requires_grad = True

    elif freeze_mode == "all":
        for p in croma.parameters():
            p.requires_grad = False

    elif freeze_mode == "substrings":
        if not trainable_substrings:
            fail("--freeze-croma-mode substrings requires --trainable-croma-substrings")

        for p in croma.parameters():
            p.requires_grad = False

        matched: List[str] = []

        for name, p in croma.named_parameters():
            if any(sub in name for sub in trainable_substrings):
                p.requires_grad = True
                matched.append(name)

        if not matched:
            warn(
                "No CROMA parameter names matched --trainable-croma-substrings. "
                "CROMA will remain fully frozen."
            )

        return {
            "freeze_mode": freeze_mode,
            "trainable_substrings": list(trainable_substrings),
            "matched_trainable_parameter_names_first_100": matched[:100],
            "matched_count": len(matched),
            "param_counts": count_parameters(croma),
        }

    else:
        fail(f"Unsupported freeze mode: {freeze_mode}")

    return {
        "freeze_mode": freeze_mode,
        "trainable_substrings": list(trainable_substrings),
        "param_counts": count_parameters(croma),
    }


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def resolve_path(path_value: Any, instance_root: Path) -> Path:
    raw = str(path_value).strip().replace("\\", "/")

    if raw == "":
        fail("Encountered empty path value in split CSV.")

    p = Path(raw)

    if p.exists():
        return p

    if not p.is_absolute():
        candidate = instance_root / raw
        if candidate.exists():
            return candidate

    fail(
        "Path does not exist:\n"
        f"  original: {raw}\n"
        f"  tried:    {path_to_str(p)}"
    )


def read_raster_window(
    raster_path: Path,
    row_off: int,
    col_off: int,
    patch_size: int,
    band_indices: Sequence[int],
    fill_value: float = 0.0,
) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        for band in band_indices:
            if band < 1 or band > src.count:
                fail(
                    f"Requested band {band}, but raster has {src.count} bands:\n"
                    f"{path_to_str(raster_path)}"
                )

        window = Window(
            col_off=int(col_off),
            row_off=int(row_off),
            width=int(patch_size),
            height=int(patch_size),
        )

        arr = src.read(
            indexes=list(band_indices),
            window=window,
            boundless=True,
            fill_value=fill_value,
            out_shape=(len(band_indices), patch_size, patch_size),
        ).astype(np.float32)

    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return arr


def read_label_window(
    label_path: Path,
    row_off: int,
    col_off: int,
    patch_size: int,
) -> np.ndarray:
    arr = read_raster_window(
        raster_path=label_path,
        row_off=row_off,
        col_off=col_off,
        patch_size=patch_size,
        band_indices=[1],
        fill_value=0.0,
    )
    arr = (arr > 0.5).astype(np.float32)
    return arr


class RawCromaSegmentationDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        instance_root: Path,
        patch_size: int = 224,
        use_manifest_band_indices: bool = True,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.instance_root = Path(instance_root)
        self.patch_size = int(patch_size)
        self.use_manifest_band_indices = bool(use_manifest_band_indices)

        if not self.split_csv.exists():
            fail(f"Split CSV does not exist:\n{path_to_str(self.split_csv)}")

        self.df = pd.read_csv(self.split_csv)

        if self.df.empty:
            fail(f"Split CSV is empty:\n{path_to_str(self.split_csv)}")

        required = [
            "patch_id",
            "optical_path",
            "sar_path",
            "label_path",
            "row_start",
            "col_start",
        ]

        missing = [c for c in required if c not in self.df.columns]
        if missing:
            fail(
                f"Split CSV missing required columns: {missing}\n"
                f"Available columns: {list(self.df.columns)}"
            )

        self.df["row_start"] = pd.to_numeric(self.df["row_start"], errors="coerce").astype(int)
        self.df["col_start"] = pd.to_numeric(self.df["col_start"], errors="coerce").astype(int)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]

        row_start = safe_int(row["row_start"])
        col_start = safe_int(row["col_start"])

        optical_path = resolve_path(row["optical_path"], self.instance_root)
        sar_path = resolve_path(row["sar_path"], self.instance_root)
        label_path = resolve_path(row["label_path"], self.instance_root)

        if self.use_manifest_band_indices and "optical_band_indices" in row.index:
            optical_bands = parse_band_indices(row["optical_band_indices"], default=list(range(1, 13)))
        else:
            optical_bands = list(range(1, 13))

        if self.use_manifest_band_indices and "sar_band_indices" in row.index:
            sar_bands = parse_band_indices(row["sar_band_indices"], default=[1, 2])
            # Safety: for SNAP-GRD we only want VV/VH, not VV_minus_VH.
            sar_bands = sar_bands[:2]
        else:
            sar_bands = [1, 2]

        optical = read_raster_window(
            raster_path=optical_path,
            row_off=row_start,
            col_off=col_start,
            patch_size=self.patch_size,
            band_indices=optical_bands,
            fill_value=0.0,
        )

        sar = read_raster_window(
            raster_path=sar_path,
            row_off=row_start,
            col_off=col_start,
            patch_size=self.patch_size,
            band_indices=sar_bands,
            fill_value=0.0,
        )

        mask = read_label_window(
            label_path=label_path,
            row_off=row_start,
            col_off=col_start,
            patch_size=self.patch_size,
        )

        if optical.shape[0] != 12:
            fail(f"Expected 12 optical bands, got {optical.shape[0]} for patch {row['patch_id']}")

        if sar.shape[0] != 2:
            fail(f"Expected 2 SAR bands, got {sar.shape[0]} for patch {row['patch_id']}")

        return {
            "optical": torch.from_numpy(optical),
            "sar": torch.from_numpy(sar),
            "mask": torch.from_numpy(mask),
            "patch_id": str(row["patch_id"]),
            "city": str(row["city"]) if "city" in row.index else "",
            "region": str(row["region"]) if "region" in row.index else "",
            "label_positive_pixels": float(row["label_positive_pixels"]) if "label_positive_pixels" in row.index else float(mask.sum()),
        }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )


# ---------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------

def normalize_like_croma_readme(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-sample, per-channel clipping + min-max scaling.

    This mirrors the normalization style used during the dense feature extraction.
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


# ---------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------

class ConvGNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 32,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        group_count = min(groups, out_channels)
        while out_channels % group_count != 0 and group_count > 1:
            group_count -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(group_count, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PyramidPoolingModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ppm_channels: int,
        bins: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()

        self.paths = nn.ModuleList(
    [
        nn.Sequential(
            nn.AdaptiveAvgPool2d(bin_size),
            # Use groups=1 here because the bin=1 PPM branch produces
            # B x C x 1 x 1. With batch size 1, GroupNorm with many groups
            # can fail because each group contains only one value.
            ConvGNReLU(in_channels, ppm_channels, kernel_size=1, groups=1),
        )
        for bin_size in bins
    ]
)

        self.bottleneck = ConvGNReLU(
            in_channels + len(bins) * ppm_channels,
            in_channels,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [x]

        for path in self.paths:
            y = path(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(y)

        return self.bottleneck(torch.cat(outs, dim=1))


class SingleScaleUPerNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        decoder_channels: int = 256,
        ppm_channels: int = 64,
        dropout: float = 0.10,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            ConvGNReLU(in_channels, decoder_channels, kernel_size=1),
            ConvGNReLU(decoder_channels, decoder_channels, kernel_size=3),
        )

        self.ppm = PyramidPoolingModule(
            in_channels=decoder_channels,
            ppm_channels=ppm_channels,
            bins=(1, 2, 3, 6),
        )

        self.up1 = nn.Sequential(
            ConvGNReLU(decoder_channels, decoder_channels // 2),
            ConvGNReLU(decoder_channels // 2, decoder_channels // 2),
        )

        self.up2 = nn.Sequential(
            ConvGNReLU(decoder_channels // 2, decoder_channels // 4),
            ConvGNReLU(decoder_channels // 4, decoder_channels // 4),
        )

        self.up3 = nn.Sequential(
            ConvGNReLU(decoder_channels // 4, decoder_channels // 8),
            ConvGNReLU(decoder_channels // 8, decoder_channels // 8),
        )

        self.head = nn.Sequential(
            nn.Dropout2d(p=float(dropout)),
            nn.Conv2d(decoder_channels // 8, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.ppm(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)

        logits = self.head(x)

        if logits.shape[-2:] != (224, 224):
            logits = F.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=False)

        return logits


# ---------------------------------------------------------------------
# CROMA + decoder model
# ---------------------------------------------------------------------

def dense_tokens_to_nchw(
    tensor: torch.Tensor,
    remove_cls_token: bool = True,
) -> torch.Tensor:
    """
    Convert CROMA dense output to B x C x H x W while preserving gradients.
    Supports:
      B x tokens x C
      B x C x tokens
      B x C x H x W
      B x H x W x C
    """
    if tensor.ndim == 4:
        b, d1, d2, d3 = tensor.shape

        if d1 in {768, 384, 512, 1024} and d2 > 1 and d3 > 1:
            return tensor

        if d3 in {768, 384, 512, 1024} and d1 > 1 and d2 > 1:
            return tensor.permute(0, 3, 1, 2).contiguous()

        fail(f"Unsupported 4D CROMA dense shape: {tuple(tensor.shape)}")

    if tensor.ndim == 3:
        b, a, c = tensor.shape

        # B x tokens x C
        if c in {768, 384, 512, 1024}:
            tokens = a

            if remove_cls_token and tokens > 1:
                possible = tokens - 1
                hw = int(round(math.sqrt(possible)))
                if hw * hw == possible:
                    tensor = tensor[:, 1:, :]
                    tokens = possible

            hw = int(round(math.sqrt(tokens)))
            if hw * hw != tokens:
                fail(f"Token count {tokens} is not a square. Shape: {tuple(tensor.shape)}")

            return tensor.reshape(b, hw, hw, c).permute(0, 3, 1, 2).contiguous()

        # B x C x tokens
        if a in {768, 384, 512, 1024}:
            channels = a
            tokens = c

            if remove_cls_token and tokens > 1:
                possible = tokens - 1
                hw = int(round(math.sqrt(possible)))
                if hw * hw == possible:
                    tensor = tensor[:, :, 1:]
                    tokens = possible

            hw = int(round(math.sqrt(tokens)))
            if hw * hw != tokens:
                fail(f"Token count {tokens} is not a square. Shape: {tuple(tensor.shape)}")

            return tensor.reshape(b, channels, hw, hw).contiguous()

        fail(f"Unsupported 3D CROMA dense shape: {tuple(tensor.shape)}")

    fail(f"Unsupported CROMA dense tensor ndim={tensor.ndim}, shape={tuple(tensor.shape)}")


class CromaUPerNetSegmentationModel(nn.Module):
    def __init__(
        self,
        croma: nn.Module,
        feature_key: str = "joint_encodings",
        decoder_channels: int = 256,
        ppm_channels: int = 64,
        dropout: float = 0.10,
        normalize_inputs: bool = True,
        remove_cls_token: bool = True,
    ) -> None:
        super().__init__()

        self.croma = croma
        self.feature_key = str(feature_key)
        self.normalize_inputs = bool(normalize_inputs)
        self.remove_cls_token = bool(remove_cls_token)

        self.decoder = SingleScaleUPerNetDecoder(
            in_channels=768,
            decoder_channels=int(decoder_channels),
            ppm_channels=int(ppm_channels),
            dropout=float(dropout),
            out_channels=1,
        )

    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
        return_dense: bool = False,
    ) -> Any:
        if self.normalize_inputs:
            sar = normalize_like_croma_readme(sar)
            optical = normalize_like_croma_readme(optical)

        croma_has_trainable_params = any(
            p.requires_grad for p in self.croma.parameters()
        )

        if croma_has_trainable_params:
            outputs = self.croma(
                SAR_images=sar,
                optical_images=optical,
            )
        else:
            # Important for 4GB GPUs:
            # if CROMA is frozen, do not store encoder activations for backward.
            self.croma.eval()
            with torch.no_grad():
                outputs = self.croma(
                    SAR_images=sar,
                    optical_images=optical,
                )

        if not isinstance(outputs, dict):
            fail(f"CROMA output must be a dict, got {type(outputs)}")

        if self.feature_key not in outputs:
            available = {
                key: tuple(value.shape) if hasattr(value, "shape") else str(type(value))
                for key, value in outputs.items()
            }
            fail(
                f"Feature key '{self.feature_key}' not found in CROMA outputs.\n"
                f"Available outputs:\n{json.dumps(jsonable(available), indent=2)}"
            )

        dense = dense_tokens_to_nchw(
            outputs[self.feature_key],
            remove_cls_token=self.remove_cls_token,
        )

        logits = self.decoder(dense)

        if return_dense:
            return logits, dense

        return logits


# ---------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        denominator = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.register_buffer(
            "pos_weight_tensor",
            torch.tensor([float(pos_weight)], dtype=torch.float32),
        )
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight_tensor.to(device=logits.device, dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
        )
        dice = self.dice(logits, targets)

        return self.bce_weight * bce + self.dice_weight * dice


@dataclass
class MetricAccumulator:
    threshold: float = 0.5
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0
    loss_sum: float = 0.0
    n_batches: int = 0
    n_pixels: float = 0.0
    pred_pos_pixels: float = 0.0
    gt_pos_pixels: float = 0.0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_value: float = 0.0,
    ) -> None:
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= float(self.threshold)).float()
            targets = (targets >= 0.5).float()

            self.tp += float((preds * targets).sum().item())
            self.fp += float((preds * (1.0 - targets)).sum().item())
            self.fn += float(((1.0 - preds) * targets).sum().item())
            self.tn += float(((1.0 - preds) * (1.0 - targets)).sum().item())
            self.loss_sum += float(loss_value)
            self.n_batches += 1
            self.n_pixels += float(targets.numel())
            self.pred_pos_pixels += float(preds.sum().item())
            self.gt_pos_pixels += float(targets.sum().item())

    def compute(self) -> Dict[str, float]:
        eps = 1e-8

        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        specificity = self.tn / (self.tn + self.fp + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        balanced_accuracy = 0.5 * (recall + specificity)

        return {
            "threshold": float(self.threshold),
            "loss": self.loss_sum / max(1, self.n_batches),
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "pred_pos_pct": float(100.0 * self.pred_pos_pixels / max(eps, self.n_pixels)),
            "gt_pos_pct": float(100.0 * self.gt_pos_pixels / max(eps, self.n_pixels)),
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "tn": float(self.tn),
        }


def compute_pos_weight_from_csv(
    train_csv: Path,
    patch_size: int,
    max_pos_weight: float,
) -> Tuple[float, Dict[str, Any]]:
    df = pd.read_csv(train_csv)

    if "label_positive_pixels" not in df.columns:
        warn("label_positive_pixels not found. Falling back to pos_weight=1.0.")
        return 1.0, {
            "method": "fallback",
            "reason": "label_positive_pixels missing",
            "pos_weight": 1.0,
        }

    pos_pixels = pd.to_numeric(df["label_positive_pixels"], errors="coerce").fillna(0).sum()
    total_pixels = int(len(df)) * int(patch_size) * int(patch_size)
    neg_pixels = total_pixels - pos_pixels

    if pos_pixels <= 0:
        warn("No positive pixels found. Falling back to pos_weight=1.0.")
        return 1.0, {
            "method": "fallback",
            "reason": "no positive pixels",
            "pos_weight": 1.0,
        }

    raw_pos_weight = float(neg_pixels / pos_pixels)
    clipped = float(min(raw_pos_weight, float(max_pos_weight)))

    return clipped, {
        "method": "label_positive_pixels",
        "train_rows": int(len(df)),
        "total_pixels": int(total_pixels),
        "positive_pixels": int(pos_pixels),
        "negative_pixels": int(neg_pixels),
        "raw_pos_weight": raw_pos_weight,
        "max_pos_weight": float(max_pos_weight),
        "pos_weight": clipped,
    }


# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------

def optimizer_parameter_groups(
    model: CromaUPerNetSegmentationModel,
    encoder_lr: float,
    decoder_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    encoder_params = [p for p in model.croma.parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]

    groups: List[Dict[str, Any]] = []

    if encoder_params:
        groups.append(
            {
                "params": encoder_params,
                "lr": float(encoder_lr),
                "weight_decay": float(weight_decay),
                "name": "encoder",
            }
        )

    if decoder_params:
        groups.append(
            {
                "params": decoder_params,
                "lr": float(decoder_lr),
                "weight_decay": float(weight_decay),
                "name": "decoder",
            }
        )

    if not groups:
        fail("No trainable parameters found.")

    return groups


def grad_summary(model: nn.Module) -> Dict[str, Any]:
    croma_grad_params = 0
    croma_grad_norm = 0.0
    decoder_grad_params = 0
    decoder_grad_norm = 0.0

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        norm = float(p.grad.detach().float().norm().item())

        if name.startswith("croma."):
            croma_grad_params += 1
            croma_grad_norm += norm
        elif name.startswith("decoder."):
            decoder_grad_params += 1
            decoder_grad_norm += norm

    return {
        "croma_grad_params": croma_grad_params,
        "croma_grad_norm_sum": croma_grad_norm,
        "decoder_grad_params": decoder_grad_params,
        "decoder_grad_norm_sum": decoder_grad_norm,
    }


def train_one_epoch(
    model: CromaUPerNetSegmentationModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_accum_steps: int,
    max_batches: Optional[int],
    max_grad_norm: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    model.train()

    acc = MetricAccumulator(threshold=0.5)
    grad_info_last: Dict[str, Any] = {}

    optimizer.zero_grad(set_to_none=True)

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=f"train epoch {epoch}", unit="batch")

    effective_batches = 0

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        sar = batch["sar"].to(device=device, dtype=torch.float32, non_blocking=True)
        optical = batch["optical"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(sar=sar, optical=optical)
                loss = criterion(logits, masks)
                loss_for_backward = loss / int(grad_accum_steps)

            if scaler is None:
                fail("AMP requested but scaler is None.")

            scaler.scale(loss_for_backward).backward()

        else:
            logits = model(sar=sar, optical=optical)
            loss = criterion(logits, masks)
            loss_for_backward = loss / int(grad_accum_steps)
            loss_for_backward.backward()

        loss_value = float(loss.detach().cpu().item())

        if not math.isfinite(loss_value):
            fail(f"Non-finite training loss at batch {batch_idx}: {loss_value}")

        acc.update(logits.detach(), masks.detach(), loss_value=loss_value)

        effective_batches += 1
        do_step = effective_batches % int(grad_accum_steps) == 0

        if do_step:
            if max_grad_norm > 0:
                if use_amp and device.type == "cuda" and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))

            grad_info_last = grad_summary(model)

            if use_amp and device.type == "cuda":
                assert scaler is not None
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        del sar, optical, masks, logits, loss

    # Handle leftover accumulated gradients.
    if effective_batches > 0 and effective_batches % int(grad_accum_steps) != 0:
        if max_grad_norm > 0:
            if use_amp and device.type == "cuda" and scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))

        grad_info_last = grad_summary(model)

        if use_amp and device.type == "cuda":
            assert scaler is not None
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

    metrics = acc.compute()
    return metrics, grad_info_last


@torch.no_grad()
def evaluate(
    model: CromaUPerNetSegmentationModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    split_name: str,
    threshold: float,
    max_batches: Optional[int],
) -> Dict[str, float]:
    model.eval()

    acc = MetricAccumulator(threshold=float(threshold))

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=f"{split_name} epoch {epoch}", unit="batch")

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        sar = batch["sar"].to(device=device, dtype=torch.float32, non_blocking=True)
        optical = batch["optical"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        logits = model(sar=sar, optical=optical)
        loss = criterion(logits, masks)
        loss_value = float(loss.detach().cpu().item())

        if not math.isfinite(loss_value):
            fail(f"Non-finite {split_name} loss at batch {batch_idx}: {loss_value}")

        acc.update(logits.detach(), masks.detach(), loss_value=loss_value)

        del sar, optical, masks, logits, loss

    return acc.compute()


@torch.no_grad()
def evaluate_thresholds(
    model: CromaUPerNetSegmentationModel,
    loader: DataLoader,
    device: torch.device,
    thresholds: Sequence[float],
    split_name: str,
    max_batches: Optional[int],
) -> List[Dict[str, float]]:
    model.eval()

    accs = [MetricAccumulator(threshold=float(t)) for t in thresholds]

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=f"{split_name} threshold sweep", unit="batch")

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        sar = batch["sar"].to(device=device, dtype=torch.float32, non_blocking=True)
        optical = batch["optical"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        logits = model(sar=sar, optical=optical)

        for acc in accs:
            acc.update(logits.detach(), masks.detach(), loss_value=0.0)

        del sar, optical, masks, logits

    return [acc.compute() for acc in accs]


def make_thresholds(start: float, end: float, step: float) -> List[float]:
    vals: List[float] = []
    x = float(start)
    while x <= float(end) + 1e-9:
        vals.append(round(x, 6))
        x += float(step)
    return vals


def select_best_threshold(rows: List[Dict[str, float]], metric: str = "iou") -> Dict[str, float]:
    if not rows:
        fail("No threshold rows available.")

    return sorted(
        rows,
        key=lambda r: (
            float(r[metric]),
            float(r["dice"]),
            -abs(float(r["pred_pos_pct"]) - float(r["gt_pos_pct"])),
        ),
        reverse=True,
    )[0]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "created_utc": now_utc(),
        },
        path,
    )


def load_model_checkpoint(path: Path, model: nn.Module, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        fail(f"Checkpoint not found:\n{path_to_str(path)}")

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    state = ckpt.get("model_state_dict")
    if state is None:
        fail(f"Checkpoint does not contain model_state_dict:\n{path_to_str(path)}")

    model.load_state_dict(state, strict=True)
    return ckpt


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    return (
        f"{prefix}: "
        f"loss={metrics['loss']:.4f}, "
        f"IoU={metrics['iou']:.4f}, "
        f"Dice={metrics['dice']:.4f}, "
        f"P={metrics['precision']:.4f}, "
        f"R={metrics['recall']:.4f}, "
        f"pred+={metrics['pred_pos_pct']:.3f}%, "
        f"gt+={metrics['gt_pos_pct']:.3f}%"
    )


# ---------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------

def save_prediction_previews(
    model: CromaUPerNetSegmentationModel,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    threshold: float,
    max_items: int,
    title_prefix: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warn(f"matplotlib not available; skipping previews: {exc}")
        return

    ensure_dir(output_dir)
    model.eval()

    saved = 0

    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device=device, dtype=torch.float32, non_blocking=True)
            optical = batch["optical"].to(device=device, dtype=torch.float32, non_blocking=True)
            masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

            logits = model(sar=sar, optical=optical)
            probs = torch.sigmoid(logits)

            b = masks.shape[0]

            for i in range(b):
                if saved >= int(max_items):
                    return

                gt = masks[i, 0].detach().cpu().numpy()
                prob = probs[i, 0].detach().cpu().numpy()
                pred = (prob >= float(threshold)).astype(np.float32)

                patch_id = batch["patch_id"][i] if isinstance(batch["patch_id"], list) else str(saved)

                fig, axes = plt.subplots(1, 3, figsize=(10, 3))
                axes[0].imshow(gt, vmin=0, vmax=1)
                axes[0].set_title("Ground truth")
                axes[0].axis("off")

                axes[1].imshow(prob, vmin=0, vmax=1)
                axes[1].set_title("Probability")
                axes[1].axis("off")

                axes[2].imshow(pred, vmin=0, vmax=1)
                axes[2].set_title(f"Prediction t={threshold:.2f}")
                axes[2].axis("off")

                fig.suptitle(f"{title_prefix} | {patch_id}", fontsize=9)
                fig.tight_layout()

                out_path = output_dir / f"sample{saved:02d}_thr{threshold:.2f}.png"
                fig.savefig(out_path, dpi=150)
                plt.close(fig)

                saved += 1

            del sar, optical, masks, logits, probs


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))

    instance_root = Path(args.instance_root)
    croma_repo = Path(args.croma_repo)
    weights_path = Path(args.weights_path)

    if args.heldout_region not in REGIONS:
        fail(f"Unsupported heldout region: {args.heldout_region}")

    if not weights_path.exists():
        fail(f"CROMA weights path does not exist:\n{path_to_str(weights_path)}")

    alignment_dir = Path(args.alignment_dir) if args.alignment_dir else default_alignment_dir(instance_root)

    train_csv = split_csv_path(alignment_dir, args.heldout_region, "train")
    val_csv = split_csv_path(alignment_dir, args.heldout_region, "val")
    test_csv = split_csv_path(alignment_dir, args.heldout_region, "test")

    training_root = Path(args.output_root) if args.output_root else default_training_root(instance_root)

    run_name = args.run_name
    if run_name is None:
        run_name = (
            f"heldout_{slug_region(args.heldout_region)}_"
            f"finetune_croma_epochs{args.epochs}_bs{args.batch_size}_"
            f"acc{args.grad_accum_steps}_posw{args.max_pos_weight}"
        )

    run_dir = training_root / run_name
    checkpoints_dir = run_dir / "checkpoints"
    previews_dir = run_dir / "previews"
    threshold_dir = run_dir / "threshold_sweep_best"

    ensure_run_dir(run_dir, overwrite=bool(args.overwrite))
    ensure_dir(checkpoints_dir)
    ensure_dir(previews_dir)
    ensure_dir(threshold_dir)

    banner("End-to-end CROMA + UPerNet fine-tuning")

    log("INFO", f"Instance root:   {path_to_str(instance_root)}")
    log("INFO", f"CROMA repo:      {path_to_str(croma_repo)}")
    log("INFO", f"Weights path:    {path_to_str(weights_path)}")
    log("INFO", f"Held-out region: {args.heldout_region}")
    log("INFO", f"Train CSV:       {path_to_str(train_csv)}")
    log("INFO", f"Val CSV:         {path_to_str(val_csv)}")
    log("INFO", f"Test CSV:        {path_to_str(test_csv)}")
    log("INFO", f"Run dir:         {path_to_str(run_dir)}")

    device = choose_device(force_cpu=bool(args.force_cpu), device_index=int(args.device_index))
    log("INFO", f"Selected device: {device}")

    pin_memory = bool(args.pin_memory) and device.type == "cuda"

    train_dataset = RawCromaSegmentationDataset(
        split_csv=train_csv,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        use_manifest_band_indices=bool(args.use_manifest_band_indices),
    )

    val_dataset = RawCromaSegmentationDataset(
        split_csv=val_csv,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        use_manifest_band_indices=bool(args.use_manifest_band_indices),
    )

    test_dataset = RawCromaSegmentationDataset(
        split_csv=test_csv,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        use_manifest_band_indices=bool(args.use_manifest_band_indices),
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )

    val_loader = make_loader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )

    test_loader = make_loader(
        test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )

    log("INFO", f"Train patches: {len(train_dataset):,}")
    log("INFO", f"Val patches:   {len(val_dataset):,}")
    log("INFO", f"Test patches:  {len(test_dataset):,}")

    pos_weight, pos_weight_info = compute_pos_weight_from_csv(
        train_csv=train_csv,
        patch_size=int(args.patch_size),
        max_pos_weight=float(args.max_pos_weight),
    )

    log("INFO", f"Training pos_weight: {pos_weight:.4f}")
    log("INFO", f"pos_weight details: {json.dumps(jsonable(pos_weight_info), ensure_ascii=False)}")

    PretrainedCROMA = import_pretrained_croma(croma_repo)

    log("STEP", "Loading PretrainedCROMA")
    croma = PretrainedCROMA(
        pretrained_path=str(weights_path),
        size=str(args.model_size),
        modality="both",
        image_resolution=int(args.image_resolution),
    )

    freeze_info = apply_croma_freezing(
        croma=croma,
        freeze_mode=str(args.freeze_croma_mode),
        trainable_substrings=args.trainable_croma_substrings,
    )

    model = CromaUPerNetSegmentationModel(
        croma=croma,
        feature_key=str(args.feature_key),
        decoder_channels=int(args.decoder_channels),
        ppm_channels=int(args.ppm_channels),
        dropout=float(args.dropout),
        normalize_inputs=(args.normalization == "per_sample_channel"),
        remove_cls_token=bool(args.remove_cls_token),
    ).to(device)

    model_counts = count_parameters(model)
    croma_counts = count_parameters(model.croma)
    decoder_counts = count_parameters(model.decoder)

    log("INFO", f"CROMA parameter counts: {croma_counts}")
    log("INFO", f"Decoder parameter counts: {decoder_counts}")
    log("INFO", f"Total model parameter counts: {model_counts}")
    log("INFO", f"Freeze info: {json.dumps(jsonable(freeze_info), ensure_ascii=False)[:2000]}")

    criterion = BCEDiceLoss(
        pos_weight=float(pos_weight),
        bce_weight=float(args.bce_weight),
        dice_weight=float(args.dice_weight),
    ).to(device)

    param_groups = optimizer_parameter_groups(
        model=model,
        encoder_lr=float(args.encoder_lr),
        decoder_lr=float(args.decoder_lr),
        weight_decay=float(args.weight_decay),
    )

    optimizer = torch.optim.AdamW(param_groups)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, int(args.lr_patience)),
    )

    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    config = {
        "created_utc": now_utc(),
        "args": vars(args),
        "run_dir": path_to_str(run_dir),
        "train_csv": path_to_str(train_csv),
        "val_csv": path_to_str(val_csv),
        "test_csv": path_to_str(test_csv),
        "train_patches": len(train_dataset),
        "val_patches": len(val_dataset),
        "test_patches": len(test_dataset),
        "pos_weight_info": pos_weight_info,
        "freeze_info": freeze_info,
        "model_counts": {
            "total": model_counts,
            "croma": croma_counts,
            "decoder": decoder_counts,
        },
    }

    write_json(run_dir / "config.json", config)

    metrics_rows: List[Dict[str, Any]] = []

    best_val_iou = -1.0
    best_epoch = -1

    started = time.time()

    try:
        for epoch in range(1, int(args.epochs) + 1):
            epoch_started = time.time()

            train_metrics, grad_info = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                use_amp=use_amp,
                scaler=scaler,
                grad_accum_steps=int(args.grad_accum_steps),
                max_batches=args.max_train_batches,
                max_grad_norm=float(args.max_grad_norm),
            )

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                split_name="val",
                threshold=0.5,
                max_batches=args.max_val_batches,
            )

            scheduler.step(val_metrics["iou"])

            epoch_seconds = round(time.time() - epoch_started, 3)

            lr_encoder = None
            lr_decoder = None
            for group in optimizer.param_groups:
                if group.get("name") == "encoder":
                    lr_encoder = float(group["lr"])
                elif group.get("name") == "decoder":
                    lr_decoder = float(group["lr"])

            row = {
                "epoch": epoch,
                "seconds": epoch_seconds,
                "lr_encoder": lr_encoder,
                "lr_decoder": lr_decoder,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"grad_{k}": v for k, v in grad_info.items()},
            }

            metrics_rows.append(row)
            write_csv(run_dir / "metrics.csv", metrics_rows)

            log("INFO", format_metrics("train", train_metrics))
            log("INFO", format_metrics("val", val_metrics))
            log("INFO", f"grad info: {json.dumps(jsonable(grad_info), ensure_ascii=False)}")
            log("INFO", f"epoch={epoch}, seconds={epoch_seconds}, lr_encoder={lr_encoder}, lr_decoder={lr_decoder}")

            save_checkpoint(
                checkpoints_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                args=args,
            )

            if val_metrics["iou"] > best_val_iou:
                best_val_iou = float(val_metrics["iou"])
                best_epoch = int(epoch)

                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=row,
                    args=args,
                )

                log("OK", f"New best checkpoint saved at epoch {epoch}: val_iou={best_val_iou:.4f}")

            clear_torch_memory()

        # Load best checkpoint before final evaluation.
        log("STEP", "Loading best checkpoint for final evaluation")
        best_ckpt = load_model_checkpoint(checkpoints_dir / "best.pt", model, device=device)
        log("INFO", f"Loaded best epoch: {best_ckpt.get('epoch')}")

        if not args.skip_threshold_sweep:
            thresholds = make_thresholds(
                float(args.threshold_start),
                float(args.threshold_end),
                float(args.threshold_step),
            )

            val_sweep_rows = evaluate_thresholds(
                model=model,
                loader=val_loader,
                device=device,
                thresholds=thresholds,
                split_name="val",
                max_batches=args.max_val_batches,
            )

            write_csv(threshold_dir / "validation_threshold_sweep.csv", val_sweep_rows)

            selected = select_best_threshold(val_sweep_rows, metric=str(args.selection_metric))
            selected_threshold = float(selected["threshold"])

            test_sweep_rows = evaluate_thresholds(
                model=model,
                loader=test_loader,
                device=device,
                thresholds=[selected_threshold],
                split_name="test",
                max_batches=args.max_test_batches,
            )

            test_selected = test_sweep_rows[0]
            write_csv(threshold_dir / "test_metrics_selected_threshold.csv", test_sweep_rows)

            log("OK", f"Selected threshold: {selected_threshold:.4f}")
            log("OK", f"Validation IoU at selected threshold: {selected['iou']:.4f}")
            log("OK", f"Test IoU at selected threshold: {test_selected['iou']:.4f}")
            log("OK", f"Test Dice at selected threshold: {test_selected['dice']:.4f}")

        else:
            selected = {}
            test_selected = evaluate(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                epoch=int(args.epochs),
                split_name="test",
                threshold=0.5,
                max_batches=args.max_test_batches,
            )
            selected_threshold = 0.5

        if int(args.preview_items) > 0:
            save_prediction_previews(
                model=model,
                loader=val_loader,
                device=device,
                output_dir=previews_dir / "val_best_selected_threshold",
                threshold=float(selected_threshold),
                max_items=int(args.preview_items),
                title_prefix=f"VAL {args.heldout_region}",
            )

            save_prediction_previews(
                model=model,
                loader=test_loader,
                device=device,
                output_dir=previews_dir / "test_best_selected_threshold",
                threshold=float(selected_threshold),
                max_items=int(args.preview_items),
                title_prefix=f"TEST {args.heldout_region}",
            )

        final_summary = {
            "status": "completed",
            "heldout_region": args.heldout_region,
            "run_dir": path_to_str(run_dir),
            "best_epoch": best_epoch,
            "best_val_iou_threshold_0_5": best_val_iou,
            "selected_threshold": selected,
            "test_metrics_selected_threshold": test_selected,
            "elapsed_seconds": round(time.time() - started, 3),
            "created_utc": now_utc(),
            "config": config,
        }

        write_json(run_dir / "final_summary.json", final_summary)

        log("OK", "Fine-tuning completed.")
        log("OK", f"Run dir: {path_to_str(run_dir)}")
        log("OK", f"Best epoch: {best_epoch}")
        log("OK", f"Best val IoU at threshold 0.5: {best_val_iou:.4f}")
        log("OK", f"Selected threshold: {selected_threshold}")
        log("OK", f"Test IoU selected threshold: {test_selected.get('iou')}")

    except Exception:
        traceback.print_exc()
        fail("Fine-tuning failed. See traceback above.")

    finally:
        clear_torch_memory()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune CROMA + UPerNet-style decoder end-to-end for LORO segmentation."
    )

    parser.add_argument("--instance-root", required=True)
    parser.add_argument("--croma-repo", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--heldout-region", required=True, choices=REGIONS)

    parser.add_argument("--alignment-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--image-resolution", type=int, default=224)
    parser.add_argument("--model-size", choices=["base", "large"], default="base")
    parser.add_argument("--feature-key", default="joint_encodings")
    parser.add_argument("--remove-cls-token", action="store_true", default=True)
    parser.add_argument("--keep-cls-token", dest="remove_cls_token", action="store_false")

    parser.add_argument(
        "--freeze-croma-mode",
        choices=["none", "all", "substrings"],
        default="none",
        help="none = fine-tune all CROMA; all = freeze CROMA; substrings = unfreeze selected parameter names only.",
    )
    parser.add_argument(
        "--trainable-croma-substrings",
        nargs="*",
        default=[],
        help="Used only with --freeze-croma-mode substrings.",
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)

    parser.add_argument("--decoder-channels", type=int, default=256)
    parser.add_argument("--ppm-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--max-pos-weight", type=float, default=10.0)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)

    parser.add_argument(
        "--normalization",
        choices=["per_sample_channel", "none"],
        default="per_sample_channel",
    )
    parser.add_argument(
        "--use-manifest-band-indices",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--ignore-manifest-band-indices",
        dest="use_manifest_band_indices",
        action="store_false",
    )

    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")

    parser.add_argument("--skip-threshold-sweep", action="store_true")
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--selection-metric",
        choices=["iou", "dice", "balanced_accuracy"],
        default="iou",
    )

    parser.add_argument("--preview-items", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())