#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
9_train_torchgeo_resnet18_upernet_s1s2_loro_224.py

Main objective
--------------
Train a lightweight, spatially pretrained segmentation model for LORO favela
segmentation using:

    S2 12 bands + SNAP-GRD S1 VV/VH

Model:
    S2 12 bands -> learnable 12-to-13 adapter
    -> TorchGeo Sentinel-2 pretrained ResNet18 feature backbone
    + lightweight S1 fusion at each ResNet feature scale
    -> UPerNet/FPN-style decoder
    -> binary favela mask

Why this script exists
----------------------
CROMA frozen dense features were trainable but generalized poorly to the
Southeast held-out region, and full CROMA fine-tuning is too heavy for a 4GB GPU.

This script tests a practical alternative:
a lighter remote-sensing pretrained backbone that can be partially fine-tuned.

Dependencies
------------
pip install timm torchgeo

Recommended smoke test
----------------------
python src/splitting_strategy_experiments/9_train_torchgeo_resnet18_upernet_s1s2_loro_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --heldout-region Southeast `
  --epochs 2 `
  --batch-size 4 `
  --max-train-batches 10 `
  --max-val-batches 5 `
  --max-test-batches 5 `
  --num-workers 0 `
  --freeze-backbone-mode all `
  --decoder-channels 128 `
  --max-pos-weight 10 `
  --run-name "smoke_torchgeo_resnet18_upernet_s1s2_Southeast_bs4" `
  --overwrite

Recommended first real run
--------------------------
python src/splitting_strategy_experiments/9_train_torchgeo_resnet18_upernet_s1s2_loro_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --heldout-region Southeast `
  --epochs 30 `
  --batch-size 8 `
  --num-workers 0 `
  --freeze-backbone-mode all `
  --decoder-channels 128 `
  --backbone-lr 1e-5 `
  --head-lr 3e-4 `
  --max-pos-weight 10 `
  --run-name "heldout_Southeast_torchgeo_resnet18_upernet_s1s2_epochs30_bs8_freezebackbone_posw10" `
  --overwrite

Optional partial fine-tuning run
--------------------------------
python src/splitting_strategy_experiments/9_train_torchgeo_resnet18_upernet_s1s2_loro_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --heldout-region Southeast `
  --epochs 30 `
  --batch-size 4 `
  --num-workers 0 `
  --freeze-backbone-mode layer4 `
  --decoder-channels 128 `
  --backbone-lr 1e-5 `
  --head-lr 3e-4 `
  --max-pos-weight 10 `
  --run-name "heldout_Southeast_torchgeo_resnet18_upernet_s1s2_epochs30_bs4_layer4_posw10" `
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
    import timm
except ImportError as exc:
    raise SystemExit(
        "[ERROR] timm is required for this script.\n"
        "Install it with:\n"
        "    pip install timm\n\n"
        f"Original error: {exc}"
    )

try:
    from torchgeo.models import ResNet18_Weights
except ImportError as exc:
    raise SystemExit(
        "[ERROR] torchgeo is required for Sentinel-2 pretrained ResNet18 weights.\n"
        "Install it with:\n"
        "    pip install torchgeo\n\n"
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


def count_parameters(module: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }


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
        / "torchgeo_resnet18_upernet_s1s2_loro_ps224_st112_cover"
    )


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
    return (arr > 0.5).astype(np.float32)


class RawS1S2SegmentationDataset(Dataset):
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
            sar_bands = sar_bands[:2]
        else:
            sar_bands = [1, 2]

        s2 = read_raster_window(
            raster_path=optical_path,
            row_off=row_start,
            col_off=col_start,
            patch_size=self.patch_size,
            band_indices=optical_bands,
            fill_value=0.0,
        )

        s1 = read_raster_window(
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

        if s2.shape[0] != 12:
            fail(f"Expected 12 S2 bands, got {s2.shape[0]} for patch {row['patch_id']}")

        if s1.shape[0] != 2:
            fail(f"Expected 2 S1 bands, got {s1.shape[0]} for patch {row['patch_id']}")

        return {
            "s2": torch.from_numpy(s2),
            "s1": torch.from_numpy(s1),
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
# Normalization
# ---------------------------------------------------------------------

def normalize_per_sample_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
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
# Model components
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


class S2To13Adapter(nn.Module):
    """
    Learnable 12-channel to 13-channel adapter.

    TorchGeo Sentinel-2 ALL pretrained ResNet18 weights expect 13 channels.
    Our processed S2 stack has 12 bands. This adapter keeps the first 12 channels
    close to identity and learns the missing/extra 13th channel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(12, 13, kernel_size=1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.proj.weight.zero_()
            for i in range(12):
                self.proj.weight[i, i, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


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
                    # groups=1 avoids GroupNorm failure for B x C x 1 x 1
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


class UPerFPNDecoder(nn.Module):
    """
    Lightweight UPerNet/FPN-style decoder.

    Input:
        list of multi-scale fused features from ResNet18:
        usually reduction 4, 8, 16, 32

    Output:
        B x 1 x 224 x 224
    """

    def __init__(
        self,
        feature_channels: Sequence[int],
        decoder_channels: int = 128,
        ppm_channels: int = 32,
        dropout: float = 0.10,
        out_channels: int = 1,
        output_size: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()

        self.output_size = tuple(output_size)
        self.n_levels = len(feature_channels)

        if self.n_levels < 2:
            fail("UPerFPNDecoder requires at least two feature levels.")

        self.lateral_convs = nn.ModuleList(
            [
                ConvGNReLU(ch, decoder_channels, kernel_size=1)
                for ch in feature_channels
            ]
        )

        self.ppm = PyramidPoolingModule(
            in_channels=decoder_channels,
            ppm_channels=ppm_channels,
            bins=(1, 2, 3, 6),
        )

        self.fpn_convs = nn.ModuleList(
            [
                ConvGNReLU(decoder_channels, decoder_channels, kernel_size=3)
                for _ in feature_channels
            ]
        )

        self.fuse = nn.Sequential(
            ConvGNReLU(decoder_channels * self.n_levels, decoder_channels, kernel_size=3),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        laterals = [
            lateral(feature)
            for lateral, feature in zip(self.lateral_convs, features)
        ]

        laterals[-1] = self.ppm(laterals[-1])

        # Top-down FPN fusion.
        for i in range(len(laterals) - 1, 0, -1):
            target_hw = laterals[i - 1].shape[-2:]
            up = F.interpolate(laterals[i], size=target_hw, mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up

        outs = [
            conv(lat)
            for conv, lat in zip(self.fpn_convs, laterals)
        ]

        highest_hw = outs[0].shape[-2:]
        outs = [
            F.interpolate(out, size=highest_hw, mode="bilinear", align_corners=False)
            for out in outs
        ]

        fused = torch.cat(outs, dim=1)
        logits = self.fuse(fused)

        if logits.shape[-2:] != self.output_size:
            logits = F.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)

        return logits


class TorchGeoResNet18S1S2UPerNet(nn.Module):
    def __init__(
        self,
        decoder_channels: int = 128,
        ppm_channels: int = 32,
        sar_fusion_channels: int = 16,
        dropout: float = 0.10,
        normalization: str = "per_sample_channel",
        use_pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3, 4),
    ) -> None:
        super().__init__()

        self.normalization = str(normalization)
        self.s2_adapter = S2To13Adapter()

        weights = ResNet18_Weights.SENTINEL2_ALL_MOCO if use_pretrained else None
        in_chans = int(weights.meta["in_chans"]) if weights is not None else 13

        self.s2_backbone = timm.create_model(
            "resnet18",
            pretrained=False,
            features_only=True,
            out_indices=tuple(out_indices),
            in_chans=in_chans,
        )

        if weights is not None:
            state = weights.get_state_dict(progress=True)
            missing, unexpected = self.s2_backbone.load_state_dict(state, strict=False)
            self.weight_loading_info = {
                "weights": "ResNet18_Weights.SENTINEL2_ALL_MOCO",
                "in_chans": in_chans,
                "missing_keys_count": len(missing),
                "unexpected_keys_count": len(unexpected),
                "missing_keys_first_20": list(missing)[:20],
                "unexpected_keys_first_20": list(unexpected)[:20],
            }
        else:
            self.weight_loading_info = {
                "weights": "none",
                "in_chans": in_chans,
            }

        feature_channels = list(self.s2_backbone.feature_info.channels())

        self.sar_projs = nn.ModuleList(
            [
                ConvGNReLU(2, int(sar_fusion_channels), kernel_size=3)
                for _ in feature_channels
            ]
        )

        self.fuse_convs = nn.ModuleList(
            [
                ConvGNReLU(ch + int(sar_fusion_channels), ch, kernel_size=1)
                for ch in feature_channels
            ]
        )

        self.decoder = UPerFPNDecoder(
            feature_channels=feature_channels,
            decoder_channels=int(decoder_channels),
            ppm_channels=int(ppm_channels),
            dropout=float(dropout),
            out_channels=1,
            output_size=(224, 224),
        )

        self.feature_channels = feature_channels

    def forward(self, s2: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        if self.normalization == "per_sample_channel":
            s2 = normalize_per_sample_channel(s2)
            s1 = normalize_per_sample_channel(s1)
        elif self.normalization == "none":
            s2 = s2.float()
            s1 = s1.float()
        else:
            fail(f"Unsupported normalization: {self.normalization}")

        s2_13 = self.s2_adapter(s2)
        s2_features = self.s2_backbone(s2_13)

        fused_features: List[torch.Tensor] = []

        for feat, sar_proj, fuse in zip(s2_features, self.sar_projs, self.fuse_convs):
            s1_resized = F.interpolate(
                s1,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            s1_feat = sar_proj(s1_resized)
            fused = fuse(torch.cat([feat, s1_feat], dim=1))
            fused_features.append(fused)

        logits = self.decoder(fused_features)
        return logits


# ---------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------

def set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(value)


def apply_backbone_freezing(
    model: TorchGeoResNet18S1S2UPerNet,
    mode: str,
) -> Dict[str, Any]:
    """
    mode:
      all: freeze S2 pretrained backbone; train S2 adapter + S1 fusion + decoder
      layer4: freeze backbone except layer4
      layer3_layer4: freeze backbone except layer3 and layer4
      none: train full S2 backbone
    """
    mode = str(mode)

    set_requires_grad(model.s2_backbone, False)

    matched: List[str] = []

    if mode == "all":
        pass

    elif mode in {"layer4", "layer3_layer4", "none"}:
        if mode == "none":
            for name, p in model.s2_backbone.named_parameters():
                p.requires_grad = True
                matched.append(name)
        else:
            trainable_prefixes = ["layer4"] if mode == "layer4" else ["layer3", "layer4"]

            for name, p in model.s2_backbone.named_parameters():
                if any(name.startswith(prefix) for prefix in trainable_prefixes):
                    p.requires_grad = True
                    matched.append(name)

    else:
        fail(f"Unsupported --freeze-backbone-mode: {mode}")

    # Always train adapter, SAR fusion, and decoder.
    set_requires_grad(model.s2_adapter, True)
    set_requires_grad(model.sar_projs, True)
    set_requires_grad(model.fuse_convs, True)
    set_requires_grad(model.decoder, True)

    return {
        "freeze_backbone_mode": mode,
        "trainable_backbone_parameter_names_first_100": matched[:100],
        "trainable_backbone_parameter_count": len(matched),
        "s2_backbone_counts": count_parameters(model.s2_backbone),
        "s2_adapter_counts": count_parameters(model.s2_adapter),
        "sar_projs_counts": count_parameters(model.sar_projs),
        "fuse_convs_counts": count_parameters(model.fuse_convs),
        "decoder_counts": count_parameters(model.decoder),
        "total_counts": count_parameters(model),
    }


def optimizer_parameter_groups(
    model: TorchGeoResNet18S1S2UPerNet,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    backbone_params = [p for p in model.s2_backbone.parameters() if p.requires_grad]

    head_modules = [
        model.s2_adapter,
        model.sar_projs,
        model.fuse_convs,
        model.decoder,
    ]

    head_params: List[nn.Parameter] = []
    for module in head_modules:
        head_params.extend([p for p in module.parameters() if p.requires_grad])

    groups: List[Dict[str, Any]] = []

    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": float(backbone_lr),
                "weight_decay": float(weight_decay),
                "name": "backbone",
            }
        )

    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": float(head_lr),
                "weight_decay": float(weight_decay),
                "name": "head",
            }
        )

    if not groups:
        fail("No trainable parameters found.")

    return groups


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

def grad_summary(model: nn.Module) -> Dict[str, Any]:
    backbone_grad_params = 0
    backbone_grad_norm = 0.0
    head_grad_params = 0
    head_grad_norm = 0.0

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        norm = float(p.grad.detach().float().norm().item())

        if name.startswith("s2_backbone."):
            backbone_grad_params += 1
            backbone_grad_norm += norm
        else:
            head_grad_params += 1
            head_grad_norm += norm

    return {
        "backbone_grad_params": backbone_grad_params,
        "backbone_grad_norm_sum": backbone_grad_norm,
        "head_grad_params": head_grad_params,
        "head_grad_norm_sum": head_grad_norm,
    }


def train_one_epoch(
    model: TorchGeoResNet18S1S2UPerNet,
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

        s2 = batch["s2"].to(device=device, dtype=torch.float32, non_blocking=True)
        s1 = batch["s1"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(s2=s2, s1=s1)
                loss = criterion(logits, masks)
                loss_for_backward = loss / int(grad_accum_steps)

            if scaler is None:
                fail("AMP requested but scaler is None.")

            scaler.scale(loss_for_backward).backward()

        else:
            logits = model(s2=s2, s1=s1)
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

        del s2, s1, masks, logits, loss

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

    return acc.compute(), grad_info_last


@torch.no_grad()
def evaluate(
    model: TorchGeoResNet18S1S2UPerNet,
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

        s2 = batch["s2"].to(device=device, dtype=torch.float32, non_blocking=True)
        s1 = batch["s1"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        logits = model(s2=s2, s1=s1)
        loss = criterion(logits, masks)
        loss_value = float(loss.detach().cpu().item())

        if not math.isfinite(loss_value):
            fail(f"Non-finite {split_name} loss at batch {batch_idx}: {loss_value}")

        acc.update(logits.detach(), masks.detach(), loss_value=loss_value)

        del s2, s1, masks, logits, loss

    return acc.compute()


@torch.no_grad()
def evaluate_thresholds(
    model: TorchGeoResNet18S1S2UPerNet,
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

        s2 = batch["s2"].to(device=device, dtype=torch.float32, non_blocking=True)
        s1 = batch["s1"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        logits = model(s2=s2, s1=s1)

        for acc in accs:
            acc.update(logits.detach(), masks.detach(), loss_value=0.0)

        del s2, s1, masks, logits

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
    model: TorchGeoResNet18S1S2UPerNet,
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
            s2 = batch["s2"].to(device=device, dtype=torch.float32, non_blocking=True)
            s1 = batch["s1"].to(device=device, dtype=torch.float32, non_blocking=True)
            masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

            logits = model(s2=s2, s1=s1)
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

            del s2, s1, masks, logits, probs


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))

    instance_root = Path(args.instance_root)

    if args.heldout_region not in REGIONS:
        fail(f"Unsupported heldout region: {args.heldout_region}")

    alignment_dir = Path(args.alignment_dir) if args.alignment_dir else default_alignment_dir(instance_root)

    train_csv = split_csv_path(alignment_dir, args.heldout_region, "train")
    val_csv = split_csv_path(alignment_dir, args.heldout_region, "val")
    test_csv = split_csv_path(alignment_dir, args.heldout_region, "test")

    training_root = Path(args.output_root) if args.output_root else default_training_root(instance_root)

    run_name = args.run_name
    if run_name is None:
        run_name = (
            f"heldout_{slug_region(args.heldout_region)}_"
            f"torchgeo_resnet18_upernet_s1s2_epochs{args.epochs}_"
            f"bs{args.batch_size}_fb{args.freeze_backbone_mode}_posw{args.max_pos_weight}"
        )

    run_dir = training_root / run_name
    checkpoints_dir = run_dir / "checkpoints"
    previews_dir = run_dir / "previews"
    threshold_dir = run_dir / "threshold_sweep_best"

    ensure_run_dir(run_dir, overwrite=bool(args.overwrite))
    ensure_dir(checkpoints_dir)
    ensure_dir(previews_dir)
    ensure_dir(threshold_dir)

    banner("TorchGeo ResNet18 + S1/S2 UPerNet LORO training")

    log("INFO", f"Instance root:       {path_to_str(instance_root)}")
    log("INFO", f"Held-out region:     {args.heldout_region}")
    log("INFO", f"Train CSV:           {path_to_str(train_csv)}")
    log("INFO", f"Val CSV:             {path_to_str(val_csv)}")
    log("INFO", f"Test CSV:            {path_to_str(test_csv)}")
    log("INFO", f"Run dir:             {path_to_str(run_dir)}")

    device = choose_device(force_cpu=bool(args.force_cpu), device_index=int(args.device_index))
    log("INFO", f"Selected device: {device}")

    pin_memory = bool(args.pin_memory) and device.type == "cuda"

    train_dataset = RawS1S2SegmentationDataset(
        split_csv=train_csv,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        use_manifest_band_indices=bool(args.use_manifest_band_indices),
    )

    val_dataset = RawS1S2SegmentationDataset(
        split_csv=val_csv,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        use_manifest_band_indices=bool(args.use_manifest_band_indices),
    )

    test_dataset = RawS1S2SegmentationDataset(
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

    model = TorchGeoResNet18S1S2UPerNet(
        decoder_channels=int(args.decoder_channels),
        ppm_channels=int(args.ppm_channels),
        sar_fusion_channels=int(args.sar_fusion_channels),
        dropout=float(args.dropout),
        normalization=str(args.normalization),
        use_pretrained=not bool(args.no_pretrained),
        out_indices=tuple(int(x) for x in args.out_indices),
    )

    freeze_info = apply_backbone_freezing(
        model=model,
        mode=str(args.freeze_backbone_mode),
    )

    model = model.to(device)

    log("INFO", f"Feature channels: {model.feature_channels}")
    log("INFO", f"Weight loading info: {json.dumps(jsonable(model.weight_loading_info), ensure_ascii=False)}")
    log("INFO", f"Freeze info: {json.dumps(jsonable(freeze_info), ensure_ascii=False)[:3000]}")
    log("INFO", f"Model parameter counts: {count_parameters(model)}")

    criterion = BCEDiceLoss(
        pos_weight=float(pos_weight),
        bce_weight=float(args.bce_weight),
        dice_weight=float(args.dice_weight),
    ).to(device)

    param_groups = optimizer_parameter_groups(
        model=model,
        backbone_lr=float(args.backbone_lr),
        head_lr=float(args.head_lr),
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
        "weight_loading_info": model.weight_loading_info,
        "feature_channels": model.feature_channels,
        "model_counts": count_parameters(model),
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

            lr_backbone = None
            lr_head = None
            for group in optimizer.param_groups:
                if group.get("name") == "backbone":
                    lr_backbone = float(group["lr"])
                elif group.get("name") == "head":
                    lr_head = float(group["lr"])

            row = {
                "epoch": epoch,
                "seconds": epoch_seconds,
                "lr_backbone": lr_backbone,
                "lr_head": lr_head,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"grad_{k}": v for k, v in grad_info.items()},
            }

            metrics_rows.append(row)
            write_csv(run_dir / "metrics.csv", metrics_rows)

            log("INFO", format_metrics("train", train_metrics))
            log("INFO", format_metrics("val", val_metrics))
            log("INFO", f"grad info: {json.dumps(jsonable(grad_info), ensure_ascii=False)}")
            log("INFO", f"epoch={epoch}, seconds={epoch_seconds}, lr_backbone={lr_backbone}, lr_head={lr_head}")

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

        # Load best checkpoint before final threshold sweep.
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
            selected_threshold = 0.5
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

        log("OK", "Training completed.")
        log("OK", f"Run dir: {path_to_str(run_dir)}")
        log("OK", f"Best epoch: {best_epoch}")
        log("OK", f"Best val IoU at threshold 0.5: {best_val_iou:.4f}")
        log("OK", f"Selected threshold: {selected_threshold}")
        log("OK", f"Test IoU selected threshold: {test_selected.get('iou')}")

    except Exception:
        traceback.print_exc()
        fail("Training failed. See traceback above.")

    finally:
        clear_torch_memory()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TorchGeo Sentinel-2 pretrained ResNet18 + S1/S2 UPerNet LORO segmentation model."
    )

    parser.add_argument("--instance-root", required=True)
    parser.add_argument("--heldout-region", required=True, choices=REGIONS)

    parser.add_argument("--alignment-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument(
        "--out-indices",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="timm ResNet feature out_indices. Default gives multi-scale features.",
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)

    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--ppm-channels", type=int, default=32)
    parser.add_argument("--sar-fusion-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument(
        "--freeze-backbone-mode",
        choices=["all", "layer4", "layer3_layer4", "none"],
        default="all",
        help="all freezes pretrained S2 ResNet; layer4 unfreezes only layer4; none fine-tunes all.",
    )

    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
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

    parser.add_argument("--no-pretrained", action="store_true", help="Disable TorchGeo pretrained weights.")

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