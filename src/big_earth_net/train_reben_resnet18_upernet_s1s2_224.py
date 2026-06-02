#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_reben_resnet18_upernet_s1s2_224.py

Main objective
--------------
Fine-tune a BigEarthNet/reBEN-pretrained ResNet18 S1+S2 encoder with a
UPerNet/FPN-style decoder for binary favela segmentation.

This script uses the train-region-covered split:

    <instance-root>/metadata/big_earth_net/region_balanced_city_split_ps224_st112_cover/

Input channel order
-------------------
The reBEN / BigEarthNet v2.0 ResNet18 S1+S2 v0.2.0 model expects 12 channels:

    [VV, VH, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]

Our S2 stack has 12 bands:

    [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]

Therefore this script reads S2 band indices:

    [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

and S1 band indices:

    [1, 2] = [VV, VH]

Main outputs
------------
Each run directory contains:

    config.json
    metrics.csv
    final_summary.json
    checkpoints/latest.pt
    checkpoints/best.pt
    figures/*.png
    threshold_sweep_best/validation_threshold_sweep.csv
    threshold_sweep_best/test_metrics_selected_threshold.csv
    threshold_sweep_best/figures/*.png

Example 10-epoch run
--------------------
python src\\big_earth_net\\train_reben_resnet18_upernet_s1s2_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --epochs 10 `
  --batch-size 8 `
  --num-workers 0 `
  --freeze-backbone-mode all `
  --decoder-channels 128 `
  --backbone-lr 1e-5 `
  --head-lr 3e-4 `
  --max-pos-weight 10 `
  --run-name "reben_resnet18_upernet_s1s2_train_region_covered_epochs30_bs8_freezebackbone_posw10" `
  --overwrite

Resume later to epoch 30
------------------------
python src\\big_earth_net\\train_reben_resnet18_upernet_s1s2_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --epochs 30 `
  --batch-size 8 `
  --num-workers 0 `
  --freeze-backbone-mode all `
  --decoder-channels 128 `
  --backbone-lr 1e-5 `
  --head-lr 3e-4 `
  --max-pos-weight 10 `
  --run-name "reben_resnet18_upernet_s1s2_train_region_covered_epochs30_bs8_freezebackbone_posw10" `
  --resume latest `
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
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "[ERROR] matplotlib is required for curve output.\n"
        "Install it with:\n"
        "    pip install matplotlib\n\n"
        f"Original error: {exc}"
    )

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
        f"Original error: {exc}"
    )

try:
    import timm
except ImportError as exc:
    raise SystemExit(
        "[ERROR] timm is required.\n"
        "Install it with:\n"
        "    pip install timm\n\n"
        f"Original error: {exc}"
    )

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "[ERROR] huggingface_hub is required.\n"
        "Install it with:\n"
        "    pip install huggingface_hub\n\n"
        f"Original error: {exc}"
    )

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError as exc:
    raise SystemExit(
        "[ERROR] safetensors is required.\n"
        "Install it with:\n"
        "    pip install safetensors\n\n"
        f"Original error: {exc}"
    )

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DEFAULT_S1_BAND_INDICES = [1, 2]
DEFAULT_S2_BAND_INDICES_REBEN = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
DEFAULT_REBEN_REPO_ID = "BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0"


# ---------------------------------------------------------------------
# Logging and utilities
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_run_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        fail(
            "Run directory already exists and is not empty:\n"
            f"{path_to_str(path)}\n\n"
            "Use --overwrite, or use --resume latest / --resume best to continue."
        )
    path.mkdir(parents=True, exist_ok=True)


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


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return path_to_str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


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


def parse_int_list(value: Optional[str], default: Sequence[int]) -> List[int]:
    if value is None or str(value).strip() == "":
        return list(default)
    parts = re.split(r"[,\s;|]+", str(value).strip())
    return [int(p) for p in parts if p]


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

def default_split_dir(instance_root: Path) -> Path:
    return (
        instance_root
        / "metadata"
        / "big_earth_net"
        / "region_balanced_city_split_ps224_st112_cover"
    )


def default_output_root(instance_root: Path) -> Path:
    return (
        instance_root
        / "experiments"
        / "big_earth_net"
        / "reben_resnet18_upernet_s1s2_train_region_covered_ps224"
    )


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def resolve_path(path_value: Any, instance_root: Path) -> Path:
    raw = str(path_value).strip().replace("\\", "/")
    if raw == "":
        fail("Encountered empty path value in CSV.")

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


class RebenS1S2SegmentationDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        instance_root: Path,
        patch_size: int,
        s1_band_indices: Sequence[int],
        s2_band_indices: Sequence[int],
    ) -> None:
        self.csv_path = Path(csv_path)
        self.instance_root = Path(instance_root)
        self.patch_size = int(patch_size)
        self.s1_band_indices = list(s1_band_indices)
        self.s2_band_indices = list(s2_band_indices)

        if not self.csv_path.exists():
            fail(f"Split CSV does not exist:\n{path_to_str(self.csv_path)}")

        self.df = pd.read_csv(self.csv_path)

        if self.df.empty:
            fail(f"Split CSV is empty:\n{path_to_str(self.csv_path)}")

        required = [
            "patch_id",
            "optical_path",
            "sar_path",
            "label_path",
            "row_start",
            "col_start",
            "city",
            "region",
        ]

        missing = [c for c in required if c not in self.df.columns]
        if missing:
            fail(
                f"Split CSV missing required columns: {missing}\n"
                f"Available columns: {list(self.df.columns)}"
            )

        self.df["row_start"] = pd.to_numeric(self.df["row_start"], errors="coerce").fillna(0).astype(int)
        self.df["col_start"] = pd.to_numeric(self.df["col_start"], errors="coerce").fillna(0).astype(int)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]

        row_start = int(row["row_start"])
        col_start = int(row["col_start"])

        optical_path = resolve_path(row["optical_path"], self.instance_root)
        sar_path = resolve_path(row["sar_path"], self.instance_root)
        label_path = resolve_path(row["label_path"], self.instance_root)

        s1 = read_raster_window(
            raster_path=sar_path,
            row_off=row_start,
            col_off=col_start,
            patch_size=self.patch_size,
            band_indices=self.s1_band_indices,
            fill_value=0.0,
        )

        s2 = read_raster_window(
            raster_path=optical_path,
            row_off=row_start,
            col_off=col_start,
            patch_size=self.patch_size,
            band_indices=self.s2_band_indices,
            fill_value=0.0,
        )

        mask = read_label_window(
            label_path=label_path,
            row_off=row_start,
            col_off=col_start,
            patch_size=self.patch_size,
        )

        if s1.shape[0] != 2:
            fail(f"Expected 2 S1 bands, got {s1.shape[0]} for patch {row['patch_id']}")

        if s2.shape[0] != 10:
            fail(f"Expected 10 reBEN S2 bands, got {s2.shape[0]} for patch {row['patch_id']}")

        # reBEN v0.2.0 S1+S2 order:
        # VV, VH, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
        x = np.concatenate([s1, s2], axis=0).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "patch_id": str(row["patch_id"]),
            "city": str(row["city"]),
            "region": str(row["region"]),
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
# Model
# ---------------------------------------------------------------------

class ConvGNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, groups: int = 32) -> None:
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
    def __init__(self, in_channels: int, ppm_channels: int, bins: Sequence[int] = (1, 2, 3, 6)) -> None:
        super().__init__()

        self.paths = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
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
            [ConvGNReLU(ch, decoder_channels, kernel_size=1) for ch in feature_channels]
        )

        self.ppm = PyramidPoolingModule(
            in_channels=decoder_channels,
            ppm_channels=ppm_channels,
            bins=(1, 2, 3, 6),
        )

        self.fpn_convs = nn.ModuleList(
            [ConvGNReLU(decoder_channels, decoder_channels, kernel_size=3) for _ in feature_channels]
        )

        self.fuse = nn.Sequential(
            ConvGNReLU(decoder_channels * self.n_levels, decoder_channels, kernel_size=3),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        laterals[-1] = self.ppm(laterals[-1])

        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + up

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        highest_hw = outs[0].shape[-2:]
        outs = [
            F.interpolate(out, size=highest_hw, mode="bilinear", align_corners=False)
            for out in outs
        ]

        fused = torch.cat(outs, dim=1)
        logits = self.fuse(fused)

        if logits.shape[-2:] != self.output_size:
            logits = F.interpolate(
                logits,
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits


class RebenResNet18UPerNet(nn.Module):
    def __init__(
        self,
        decoder_channels: int = 128,
        ppm_channels: int = 32,
        dropout: float = 0.10,
        normalization: str = "per_sample_channel",
        out_indices: Sequence[int] = (1, 2, 3, 4),
    ) -> None:
        super().__init__()

        self.normalization = str(normalization)

        self.backbone = timm.create_model(
            "resnet18",
            pretrained=False,
            features_only=True,
            out_indices=tuple(out_indices),
            in_chans=12,
        )

        feature_channels = list(self.backbone.feature_info.channels())

        self.decoder = UPerFPNDecoder(
            feature_channels=feature_channels,
            decoder_channels=int(decoder_channels),
            ppm_channels=int(ppm_channels),
            dropout=float(dropout),
            out_channels=1,
            output_size=(224, 224),
        )

        self.feature_channels = feature_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization == "per_sample_channel":
            x = normalize_per_sample_channel(x)
        elif self.normalization == "none":
            x = x.float()
        else:
            fail(f"Unsupported normalization: {self.normalization}")

        feats = self.backbone(x)
        logits = self.decoder(feats)
        return logits


# ---------------------------------------------------------------------
# reBEN weight loading
# ---------------------------------------------------------------------

def strip_common_prefixes(key: str) -> str:
    prefixes = [
        "module.",
        "model.",
        "net.",
        "classifier.",
        "image_classifier.",
        "image_model.",
        "encoder.",
        "backbone.",
        "model.model.",
        "model.backbone.",
        "model.encoder.",
        "network.",
    ]

    changed = True
    out = key

    while changed:
        changed = False
        for p in prefixes:
            if out.startswith(p):
                out = out[len(p):]
                changed = True

    return out


def load_reben_resnet18_weights(
    backbone: nn.Module,
    repo_id: str,
    local_dir: Optional[str],
    min_loaded_keys: int,
    allow_partial: bool,
) -> Dict[str, Any]:
    if local_dir:
        model_dir = Path(local_dir)
    else:
        log("INFO", f"Downloading/loading Hugging Face repo: {repo_id}")
        model_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                local_files_only=False,
            )
        )

    safetensors_path = model_dir / "model.safetensors"

    if not safetensors_path.exists():
        candidates = sorted(model_dir.glob("*.safetensors"))
        if not candidates:
            fail(f"No .safetensors file found in:\n{path_to_str(model_dir)}")
        safetensors_path = candidates[0]

    log("INFO", f"Loading reBEN weights from:\n{path_to_str(safetensors_path)}")
    ckpt = load_safetensors(str(safetensors_path))

    backbone_state = backbone.state_dict()
    new_state = dict(backbone_state)

    ckpt_keys = list(ckpt.keys())
    ckpt_by_stripped: Dict[str, List[str]] = {}

    for ck in ckpt_keys:
        stripped = strip_common_prefixes(ck)
        ckpt_by_stripped.setdefault(stripped, []).append(ck)

    matched: List[Dict[str, Any]] = []
    unmatched_model_keys: List[str] = []
    used_ckpt_keys: set[str] = set()

    for model_key, model_tensor in backbone_state.items():
        candidates: List[str] = []

        if model_key in ckpt:
            candidates.append(model_key)

        if model_key in ckpt_by_stripped:
            candidates.extend(ckpt_by_stripped[model_key])

        suffix = "." + model_key
        candidates.extend([ck for ck in ckpt_keys if ck.endswith(suffix)])

        good = []
        for ck in candidates:
            if ck in ckpt and tuple(ckpt[ck].shape) == tuple(model_tensor.shape):
                good.append(ck)

        good = list(dict.fromkeys(good))

        if len(good) >= 1:
            ck = sorted(good, key=len)[0]
            new_state[model_key] = ckpt[ck]
            used_ckpt_keys.add(ck)
            matched.append(
                {
                    "model_key": model_key,
                    "checkpoint_key": ck,
                    "shape": list(model_tensor.shape),
                }
            )
        else:
            unmatched_model_keys.append(model_key)

    backbone.load_state_dict(new_state, strict=True)

    info = {
        "repo_id": repo_id,
        "model_dir": path_to_str(model_dir),
        "safetensors_path": path_to_str(safetensors_path),
        "checkpoint_key_count": len(ckpt_keys),
        "backbone_key_count": len(backbone_state),
        "loaded_backbone_key_count": len(matched),
        "unmatched_backbone_key_count": len(unmatched_model_keys),
        "matched_first_50": matched[:50],
        "unmatched_first_50": unmatched_model_keys[:50],
        "unused_checkpoint_key_count": len(set(ckpt_keys) - used_ckpt_keys),
        "unused_checkpoint_keys_first_50": sorted(list(set(ckpt_keys) - used_ckpt_keys))[:50],
    }

    log("INFO", f"Loaded backbone keys: {len(matched)} / {len(backbone_state)}")

    if len(matched) < int(min_loaded_keys):
        message = (
            f"Only loaded {len(matched)} backbone keys, which is below "
            f"--min-loaded-backbone-keys={min_loaded_keys}.\n\n"
            "This may mean the checkpoint key format is different from expected.\n"
            "See config.json in the run directory for key diagnostics."
        )

        if allow_partial:
            warn(message)
        else:
            fail(message)

    return info


# ---------------------------------------------------------------------
# Freezing and optimizer
# ---------------------------------------------------------------------

def set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(value)


def apply_freezing(model: RebenResNet18UPerNet, mode: str) -> Dict[str, Any]:
    mode = str(mode)

    set_requires_grad(model.backbone, False)
    matched = []

    if mode == "all":
        pass

    elif mode == "layer4":
        for name, p in model.backbone.named_parameters():
            if name.startswith("layer4") or ".layer4." in name:
                p.requires_grad = True
                matched.append(name)

    elif mode == "layer3_layer4":
        for name, p in model.backbone.named_parameters():
            if (
                name.startswith("layer3")
                or name.startswith("layer4")
                or ".layer3." in name
                or ".layer4." in name
            ):
                p.requires_grad = True
                matched.append(name)

    elif mode == "none":
        for name, p in model.backbone.named_parameters():
            p.requires_grad = True
            matched.append(name)

    else:
        fail(f"Unsupported freeze mode: {mode}")

    set_requires_grad(model.decoder, True)

    return {
        "freeze_backbone_mode": mode,
        "trainable_backbone_parameter_names_first_100": matched[:100],
        "trainable_backbone_parameter_tensor_count": len(matched),
        "backbone_counts": count_parameters(model.backbone),
        "decoder_counts": count_parameters(model.decoder),
        "total_counts": count_parameters(model),
    }


def optimizer_parameter_groups(
    model: RebenResNet18UPerNet,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.decoder.parameters() if p.requires_grad]

    groups = []

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
    def __init__(self, pos_weight: float, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.register_buffer("pos_weight_tensor", torch.tensor([float(pos_weight)], dtype=torch.float32))
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

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss_value: float = 0.0) -> None:
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

        tp = float(self.tp)
        fp = float(self.fp)
        fn = float(self.fn)
        tn = float(self.tn)

        # Class 1: favela
        precision_favela = tp / (tp + fp + eps)
        recall_favela = tp / (tp + fn + eps)
        iou_favela = tp / (tp + fp + fn + eps)
        f1_favela = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        dice_favela = f1_favela
        support_favela = tp + fn

        # Class 0: no-favela/background
        # If no-favela is treated as the positive class:
        # TP0 = TN, FP0 = FN, FN0 = FP, TN0 = TP
        tp_no_favela = tn
        fp_no_favela = fn
        fn_no_favela = fp
        tn_no_favela = tp

        precision_no_favela = tp_no_favela / (tp_no_favela + fp_no_favela + eps)
        recall_no_favela = tp_no_favela / (tp_no_favela + fn_no_favela + eps)
        iou_no_favela = tp_no_favela / (tp_no_favela + fp_no_favela + fn_no_favela + eps)
        f1_no_favela = (
            2.0 * tp_no_favela
        ) / (
            2.0 * tp_no_favela + fp_no_favela + fn_no_favela + eps
        )
        dice_no_favela = f1_no_favela
        support_no_favela = tn + fp

        accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

        # Backward-compatible aliases: these are favela-class metrics.
        precision = precision_favela
        recall = recall_favela
        iou = iou_favela
        dice = dice_favela
        specificity = recall_no_favela
        balanced_accuracy = 0.5 * (recall_favela + recall_no_favela)

        macro_precision = 0.5 * (precision_favela + precision_no_favela)
        macro_recall = 0.5 * (recall_favela + recall_no_favela)
        macro_iou = 0.5 * (iou_favela + iou_no_favela)
        macro_f1 = 0.5 * (f1_favela + f1_no_favela)
        macro_dice = macro_f1

        total_support = support_favela + support_no_favela + eps

        weighted_precision = (
            precision_favela * support_favela
            + precision_no_favela * support_no_favela
        ) / total_support

        weighted_recall = (
            recall_favela * support_favela
            + recall_no_favela * support_no_favela
        ) / total_support

        weighted_iou = (
            iou_favela * support_favela
            + iou_no_favela * support_no_favela
        ) / total_support

        weighted_f1 = (
            f1_favela * support_favela
            + f1_no_favela * support_no_favela
        ) / total_support

        weighted_dice = weighted_f1

        return {
            "threshold": float(self.threshold),
            "loss": self.loss_sum / max(1, self.n_batches),

            # Backward-compatible main metrics. These are favela-class metrics.
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),

            # Favela class metrics.
            "precision_favela": float(precision_favela),
            "recall_favela": float(recall_favela),
            "iou_favela": float(iou_favela),
            "f1_favela": float(f1_favela),
            "dice_favela": float(dice_favela),
            "support_favela_pixels": float(support_favela),

            # No-favela/background class metrics.
            "precision_no_favela": float(precision_no_favela),
            "recall_no_favela": float(recall_no_favela),
            "iou_no_favela": float(iou_no_favela),
            "f1_no_favela": float(f1_no_favela),
            "dice_no_favela": float(dice_no_favela),
            "support_no_favela_pixels": float(support_no_favela),

            # Macro metrics.
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_iou": float(macro_iou),
            "macro_f1": float(macro_f1),
            "macro_dice": float(macro_dice),

            # Weighted metrics.
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_iou": float(weighted_iou),
            "weighted_f1": float(weighted_f1),
            "weighted_dice": float(weighted_dice),

            # Area diagnostics.
            "pred_pos_pct": float(100.0 * self.pred_pos_pixels / max(eps, self.n_pixels)),
            "gt_pos_pct": float(100.0 * self.gt_pos_pixels / max(eps, self.n_pixels)),
            "pred_no_favela_pct": float(
                100.0 * (self.n_pixels - self.pred_pos_pixels) / max(eps, self.n_pixels)
            ),
            "gt_no_favela_pct": float(
                100.0 * (self.n_pixels - self.gt_pos_pixels) / max(eps, self.n_pixels)
            ),

            # Raw confusion matrix counts.
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "tn": float(tn),

            # Explicit class confusion counts.
            "tp_favela": float(tp),
            "fp_favela": float(fp),
            "fn_favela": float(fn),
            "tn_favela": float(tn),

            "tp_no_favela": float(tp_no_favela),
            "fp_no_favela": float(fp_no_favela),
            "fn_no_favela": float(fn_no_favela),
            "tn_no_favela": float(tn_no_favela),
        }


def compute_pos_weight_from_csv(train_csv: Path, patch_size: int, max_pos_weight: float) -> Tuple[float, Dict[str, Any]]:
    df = pd.read_csv(train_csv)

    if "label_positive_pixels" not in df.columns:
        warn("label_positive_pixels missing. Falling back to pos_weight=1.0.")
        return 1.0, {"method": "fallback", "reason": "missing label_positive_pixels"}

    pos = pd.to_numeric(df["label_positive_pixels"], errors="coerce").fillna(0).sum()
    total = int(len(df)) * int(patch_size) * int(patch_size)
    neg = total - pos

    if pos <= 0:
        warn("No positive pixels found. Falling back to pos_weight=1.0.")
        return 1.0, {"method": "fallback", "reason": "no positives"}

    raw = float(neg / pos)
    clipped = float(min(raw, float(max_pos_weight)))

    return clipped, {
        "method": "label_positive_pixels",
        "train_rows": int(len(df)),
        "total_pixels": int(total),
        "positive_pixels": int(pos),
        "negative_pixels": int(neg),
        "raw_pos_weight": raw,
        "max_pos_weight": float(max_pos_weight),
        "pos_weight": clipped,
    }


# ---------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------

def _plot_metric_curve(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Sequence[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    available = [c for c in y_cols if c in df.columns]

    if not available:
        warn(f"Skipping plot because none of these columns exist: {y_cols}")
        return

    ensure_dir(output_path.parent)

    plt.figure(figsize=(9, 5))

    for col in available:
        plt.plot(df[x_col], df[col], marker="o", linewidth=1.5, label=col)

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_epoch_learning_curves(metrics_csv: Path, figures_dir: Path) -> None:
    if not metrics_csv.exists():
        warn(f"Cannot save learning curves because metrics CSV does not exist:\n{path_to_str(metrics_csv)}")
        return

    try:
        df = pd.read_csv(metrics_csv)
    except Exception as exc:
        warn(f"Could not read metrics CSV for plotting: {exc}")
        return

    if df.empty or "epoch" not in df.columns:
        warn("Metrics CSV is empty or missing epoch column. Skipping learning-curve plots.")
        return

    ensure_dir(figures_dir)

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_loss", "val_loss"],
        title="Training and validation loss",
        ylabel="Loss",
        output_path=figures_dir / "curve_loss.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_iou_favela", "val_iou_favela"],
        title="Training and validation favela IoU",
        ylabel="IoU",
        output_path=figures_dir / "curve_iou_favela.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_dice_favela", "val_dice_favela"],
        title="Training and validation favela Dice/F1",
        ylabel="Dice / F1",
        output_path=figures_dir / "curve_dice_favela.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_precision_favela", "val_precision_favela"],
        title="Training and validation favela precision",
        ylabel="Precision",
        output_path=figures_dir / "curve_precision_favela.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_recall_favela", "val_recall_favela"],
        title="Training and validation favela recall",
        ylabel="Recall",
        output_path=figures_dir / "curve_recall_favela.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=[
            "train_precision_favela",
            "val_precision_favela",
            "train_precision_no_favela",
            "val_precision_no_favela",
        ],
        title="Per-class precision",
        ylabel="Precision",
        output_path=figures_dir / "curve_per_class_precision.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=[
            "train_recall_favela",
            "val_recall_favela",
            "train_recall_no_favela",
            "val_recall_no_favela",
        ],
        title="Per-class recall",
        ylabel="Recall",
        output_path=figures_dir / "curve_per_class_recall.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=[
            "train_iou_favela",
            "val_iou_favela",
            "train_iou_no_favela",
            "val_iou_no_favela",
        ],
        title="Per-class IoU",
        ylabel="IoU",
        output_path=figures_dir / "curve_per_class_iou.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=[
            "train_f1_favela",
            "val_f1_favela",
            "train_f1_no_favela",
            "val_f1_no_favela",
        ],
        title="Per-class F1 / Dice",
        ylabel="F1 / Dice",
        output_path=figures_dir / "curve_per_class_f1_dice.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_macro_iou", "val_macro_iou"],
        title="Macro IoU",
        ylabel="Macro IoU",
        output_path=figures_dir / "curve_macro_iou.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_macro_f1", "val_macro_f1"],
        title="Macro F1",
        ylabel="Macro F1",
        output_path=figures_dir / "curve_macro_f1.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_macro_precision", "val_macro_precision"],
        title="Macro precision",
        ylabel="Macro precision",
        output_path=figures_dir / "curve_macro_precision.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_macro_recall", "val_macro_recall"],
        title="Macro recall",
        ylabel="Macro recall",
        output_path=figures_dir / "curve_macro_recall.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_weighted_iou", "val_weighted_iou"],
        title="Weighted IoU",
        ylabel="Weighted IoU",
        output_path=figures_dir / "curve_weighted_iou.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_weighted_f1", "val_weighted_f1"],
        title="Weighted F1",
        ylabel="Weighted F1",
        output_path=figures_dir / "curve_weighted_f1.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["train_pred_pos_pct", "train_gt_pos_pct", "val_pred_pos_pct", "val_gt_pos_pct"],
        title="Predicted vs ground-truth favela pixel percentage",
        ylabel="Favela pixels (%)",
        output_path=figures_dir / "curve_favela_pixel_percent.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=[
            "train_pred_no_favela_pct",
            "train_gt_no_favela_pct",
            "val_pred_no_favela_pct",
            "val_gt_no_favela_pct",
        ],
        title="Predicted vs ground-truth no-favela pixel percentage",
        ylabel="No-favela pixels (%)",
        output_path=figures_dir / "curve_no_favela_pixel_percent.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="epoch",
        y_cols=["lr_backbone", "lr_head"],
        title="Learning rates",
        ylabel="Learning rate",
        output_path=figures_dir / "curve_learning_rates.png",
    )


def save_threshold_sweep_curves(
    validation_sweep_csv: Path,
    figures_dir: Path,
) -> None:
    if not validation_sweep_csv.exists():
        warn(f"Cannot save threshold curves because file does not exist:\n{path_to_str(validation_sweep_csv)}")
        return

    try:
        df = pd.read_csv(validation_sweep_csv)
    except Exception as exc:
        warn(f"Could not read validation threshold sweep CSV: {exc}")
        return

    if df.empty or "threshold" not in df.columns:
        warn("Threshold sweep CSV is empty or missing threshold column. Skipping threshold plots.")
        return

    ensure_dir(figures_dir)

    _plot_metric_curve(
        df=df,
        x_col="threshold",
        y_cols=["iou_favela", "dice_favela"],
        title="Validation favela IoU and Dice by threshold",
        ylabel="Score",
        output_path=figures_dir / "threshold_favela_iou_dice.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="threshold",
        y_cols=["precision_favela", "recall_favela"],
        title="Validation favela precision and recall by threshold",
        ylabel="Score",
        output_path=figures_dir / "threshold_favela_precision_recall_vs_threshold.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="threshold",
        y_cols=["precision_favela", "precision_no_favela"],
        title="Validation per-class precision by threshold",
        ylabel="Precision",
        output_path=figures_dir / "threshold_per_class_precision.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="threshold",
        y_cols=["recall_favela", "recall_no_favela"],
        title="Validation per-class recall by threshold",
        ylabel="Recall",
        output_path=figures_dir / "threshold_per_class_recall.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="threshold",
        y_cols=["iou_favela", "iou_no_favela"],
        title="Validation per-class IoU by threshold",
        ylabel="IoU",
        output_path=figures_dir / "threshold_per_class_iou.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="threshold",
        y_cols=["f1_favela", "f1_no_favela"],
        title="Validation per-class F1/Dice by threshold",
        ylabel="F1 / Dice",
        output_path=figures_dir / "threshold_per_class_f1_dice.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="threshold",
        y_cols=["macro_iou", "macro_f1"],
        title="Validation macro IoU and macro F1 by threshold",
        ylabel="Score",
        output_path=figures_dir / "threshold_macro_iou_f1.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="threshold",
        y_cols=["macro_precision", "macro_recall"],
        title="Validation macro precision and macro recall by threshold",
        ylabel="Score",
        output_path=figures_dir / "threshold_macro_precision_recall.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="threshold",
        y_cols=["pred_pos_pct", "gt_pos_pct"],
        title="Validation predicted vs ground-truth favela percentage by threshold",
        ylabel="Favela pixels (%)",
        output_path=figures_dir / "threshold_favela_pixel_percent.png",
    )

    _plot_metric_curve(
        df=df,
        x_col="threshold",
        y_cols=["pred_no_favela_pct", "gt_no_favela_pct"],
        title="Validation predicted vs ground-truth no-favela percentage by threshold",
        ylabel="No-favela pixels (%)",
        output_path=figures_dir / "threshold_no_favela_pixel_percent.png",
    )

    if "precision_favela" in df.columns and "recall_favela" in df.columns:
        plt.figure(figsize=(6, 6))
        plt.plot(df["recall_favela"], df["precision_favela"], marker="o", linewidth=1.5)
        plt.title("Validation precision-recall curve: favela class")
        plt.xlabel("Recall favela")
        plt.ylabel("Precision favela")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "threshold_precision_recall_curve_favela.png", dpi=160)
        plt.close()

    if "precision_no_favela" in df.columns and "recall_no_favela" in df.columns:
        plt.figure(figsize=(6, 6))
        plt.plot(df["recall_no_favela"], df["precision_no_favela"], marker="o", linewidth=1.5)
        plt.title("Validation precision-recall curve: no-favela class")
        plt.xlabel("Recall no-favela")
        plt.ylabel("Precision no-favela")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "threshold_precision_recall_curve_no_favela.png", dpi=160)
        plt.close()


# ---------------------------------------------------------------------
# Training and evaluation
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

        if name.startswith("backbone."):
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
    model: RebenResNet18UPerNet,
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

        x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, masks)
                loss_for_backward = loss / int(grad_accum_steps)

            assert scaler is not None
            scaler.scale(loss_for_backward).backward()
        else:
            logits = model(x)
            loss = criterion(logits, masks)
            loss_for_backward = loss / int(grad_accum_steps)
            loss_for_backward.backward()

        loss_value = float(loss.detach().cpu().item())

        if not math.isfinite(loss_value):
            fail(f"Non-finite training loss at batch {batch_idx}: {loss_value}")

        acc.update(logits.detach(), masks.detach(), loss_value=loss_value)

        effective_batches += 1

        if effective_batches % int(grad_accum_steps) == 0:
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

        del x, masks, logits, loss

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
    model: RebenResNet18UPerNet,
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

        x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, masks)
        loss_value = float(loss.detach().cpu().item())

        acc.update(logits.detach(), masks.detach(), loss_value=loss_value)

        del x, masks, logits, loss

    return acc.compute()


@torch.no_grad()
def evaluate_thresholds(
    model: RebenResNet18UPerNet,
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

        x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        logits = model(x)

        for acc in accs:
            acc.update(logits.detach(), masks.detach(), loss_value=0.0)

        del x, masks, logits

    return [acc.compute() for acc in accs]


def make_thresholds(start: float, end: float, step: float) -> List[float]:
    vals = []
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
            float(r.get("dice", r.get("f1_favela", 0.0))),
            -abs(float(r["pred_pos_pct"]) - float(r["gt_pos_pct"])),
        ),
        reverse=True,
    )[0]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
    best_val_iou: float,
    best_epoch: int,
) -> None:
    ensure_dir(path.parent)

    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "best_val_iou": float(best_val_iou),
        "best_epoch": int(best_epoch),
        "created_utc": now_utc(),
    }

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    torch.save(payload, path)


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


def resolve_resume_checkpoint(run_dir: Path, resume: Optional[str]) -> Optional[Path]:
    if resume is None or str(resume).strip() == "":
        return None

    resume_text = str(resume).strip()

    if resume_text.lower() == "latest":
        return run_dir / "checkpoints" / "latest.pt"

    if resume_text.lower() == "best":
        return run_dir / "checkpoints" / "best.pt"

    return Path(resume_text)


def load_resume_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        fail(f"Resume checkpoint does not exist:\n{path_to_str(checkpoint_path)}")

    log("STEP", f"Resuming from checkpoint:\n{path_to_str(checkpoint_path)}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in ckpt:
        fail(f"Checkpoint does not contain model_state_dict:\n{path_to_str(checkpoint_path)}")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        warn("Resume checkpoint does not contain optimizer_state_dict. Optimizer will restart.")

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    elif scheduler is not None:
        warn("Resume checkpoint does not contain scheduler_state_dict. Scheduler will restart.")

    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return ckpt


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    return (
        f"{prefix}: "
        f"loss={metrics['loss']:.4f}, "
        f"IoU_favela={metrics['iou_favela']:.4f}, "
        f"Dice_favela={metrics['dice_favela']:.4f}, "
        f"P_favela={metrics['precision_favela']:.4f}, "
        f"R_favela={metrics['recall_favela']:.4f}, "
        f"IoU_no_favela={metrics['iou_no_favela']:.4f}, "
        f"MacroIoU={metrics['macro_iou']:.4f}, "
        f"pred+={metrics['pred_pos_pct']:.3f}%, "
        f"gt+={metrics['gt_pos_pct']:.3f}%"
    )


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))

    instance_root = Path(args.instance_root)
    split_dir = Path(args.split_dir) if args.split_dir else default_split_dir(instance_root)
    output_root = Path(args.output_root) if args.output_root else default_output_root(instance_root)

    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"

    run_name = args.run_name
    if run_name is None:
        run_name = (
            f"reben_resnet18_upernet_s1s2_epochs{args.epochs}_"
            f"bs{args.batch_size}_fb{args.freeze_backbone_mode}_posw{args.max_pos_weight}"
        )

    run_dir = output_root / run_name
    checkpoints_dir = run_dir / "checkpoints"
    threshold_dir = run_dir / "threshold_sweep_best"
    figures_dir = run_dir / "figures"

    resume_checkpoint_path = resolve_resume_checkpoint(run_dir, args.resume)

    ensure_run_dir(
        run_dir,
        overwrite=bool(args.overwrite) or resume_checkpoint_path is not None,
    )
    ensure_dir(checkpoints_dir)
    ensure_dir(threshold_dir)
    ensure_dir(figures_dir)

    banner("Train reBEN ResNet18 S1+S2 + UPerNet for favela segmentation")

    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Split dir:     {path_to_str(split_dir)}")
    log("INFO", f"Train CSV:     {path_to_str(train_csv)}")
    log("INFO", f"Val CSV:       {path_to_str(val_csv)}")
    log("INFO", f"Test CSV:      {path_to_str(test_csv)}")
    log("INFO", f"Run dir:       {path_to_str(run_dir)}")
    log("INFO", f"Resume:        {path_to_str(resume_checkpoint_path) if resume_checkpoint_path else 'none'}")

    device = choose_device(force_cpu=bool(args.force_cpu), device_index=int(args.device_index))
    log("INFO", f"Selected device: {device}")

    s1_band_indices = parse_int_list(args.s1_band_indices, DEFAULT_S1_BAND_INDICES)
    s2_band_indices = parse_int_list(args.s2_band_indices, DEFAULT_S2_BAND_INDICES_REBEN)

    log("INFO", f"S1 band indices: {s1_band_indices}")
    log("INFO", f"S2 band indices: {s2_band_indices}")
    log("INFO", "Input order: [VV, VH, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]")

    pin_memory = bool(args.pin_memory) and device.type == "cuda"

    train_dataset = RebenS1S2SegmentationDataset(
        csv_path=train_csv,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        s1_band_indices=s1_band_indices,
        s2_band_indices=s2_band_indices,
    )

    val_dataset = RebenS1S2SegmentationDataset(
        csv_path=val_csv,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        s1_band_indices=s1_band_indices,
        s2_band_indices=s2_band_indices,
    )

    test_dataset = RebenS1S2SegmentationDataset(
        csv_path=test_csv,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        s1_band_indices=s1_band_indices,
        s2_band_indices=s2_band_indices,
    )

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, pin_memory)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, pin_memory)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, pin_memory)

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

    model = RebenResNet18UPerNet(
        decoder_channels=int(args.decoder_channels),
        ppm_channels=int(args.ppm_channels),
        dropout=float(args.dropout),
        normalization=str(args.normalization),
        out_indices=tuple(int(x) for x in args.out_indices),
    )

    pretrained_info = {"enabled": False}

    if not args.no_pretrained:
        pretrained_info = load_reben_resnet18_weights(
            backbone=model.backbone,
            repo_id=str(args.reben_repo_id),
            local_dir=args.reben_local_dir,
            min_loaded_keys=int(args.min_loaded_backbone_keys),
            allow_partial=bool(args.allow_partial_pretrained),
        )
    else:
        warn("Running without reBEN pretrained weights because --no-pretrained was set.")

    freeze_info = apply_freezing(model, mode=str(args.freeze_backbone_mode))
    model = model.to(device)

    log("INFO", f"Feature channels: {model.feature_channels}")
    log("INFO", f"Pretrained loading info: {json.dumps(jsonable(pretrained_info), ensure_ascii=False)[:3000]}")
    log("INFO", f"Freeze info: {json.dumps(jsonable(freeze_info), ensure_ascii=False)[:3000]}")
    log("INFO", f"Model parameter counts: {count_parameters(model)}")

    criterion = BCEDiceLoss(
        pos_weight=float(pos_weight),
        bce_weight=float(args.bce_weight),
        dice_weight=float(args.dice_weight),
    ).to(device)

    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            model=model,
            backbone_lr=float(args.backbone_lr),
            head_lr=float(args.head_lr),
            weight_decay=float(args.weight_decay),
        )
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, int(args.lr_patience)),
    )

    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    start_epoch = 1
    best_val_iou = -1.0
    best_epoch = -1

    resume_info: Dict[str, Any] = {
        "enabled": False,
        "checkpoint_path": None,
    }

    if resume_checkpoint_path is not None:
        ckpt = load_resume_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

        resumed_epoch = int(ckpt.get("epoch", 0))
        start_epoch = resumed_epoch + 1
        best_val_iou = float(ckpt.get("best_val_iou", -1.0))
        best_epoch = int(ckpt.get("best_epoch", -1))

        resume_info = {
            "enabled": True,
            "checkpoint_path": path_to_str(resume_checkpoint_path),
            "resumed_epoch": resumed_epoch,
            "start_epoch": start_epoch,
            "best_val_iou": best_val_iou,
            "best_epoch": best_epoch,
        }

        log("OK", f"Resumed from epoch {resumed_epoch}. Continuing at epoch {start_epoch}.")
        log("OK", f"Current best epoch: {best_epoch}, best val IoU: {best_val_iou:.4f}")

    config = {
        "created_utc": now_utc(),
        "args": vars(args),
        "resume_info": resume_info,
        "run_dir": path_to_str(run_dir),
        "train_csv": path_to_str(train_csv),
        "val_csv": path_to_str(val_csv),
        "test_csv": path_to_str(test_csv),
        "train_patches": len(train_dataset),
        "val_patches": len(val_dataset),
        "test_patches": len(test_dataset),
        "s1_band_indices": s1_band_indices,
        "s2_band_indices": s2_band_indices,
        "reben_input_order": ["VV", "VH", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
        "pos_weight_info": pos_weight_info,
        "pretrained_info": pretrained_info,
        "freeze_info": freeze_info,
        "feature_channels": model.feature_channels,
        "model_counts": count_parameters(model),
    }

    write_json(run_dir / "config.json", config)

    metrics_csv_path = run_dir / "metrics.csv"

    if resume_info["enabled"] and metrics_csv_path.exists():
        previous_metrics_df = pd.read_csv(metrics_csv_path)
        metrics_rows: List[Dict[str, Any]] = previous_metrics_df.to_dict(orient="records")
        log("INFO", f"Loaded {len(metrics_rows)} previous metric rows from metrics.csv")
    else:
        metrics_rows = []

    started = time.time()

    if start_epoch > int(args.epochs):
        warn(
            f"start_epoch={start_epoch} is greater than --epochs={args.epochs}. "
            "No training epochs will run. Increase --epochs if you want to continue training."
        )

    try:
        for epoch in range(start_epoch, int(args.epochs) + 1):
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

            scheduler.step(val_metrics["iou_favela"])

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

            is_best = val_metrics["iou_favela"] > best_val_iou

            if is_best:
                best_val_iou = float(val_metrics["iou_favela"])
                best_epoch = int(epoch)

            metrics_rows.append(row)
            write_csv(metrics_csv_path, metrics_rows)

            save_epoch_learning_curves(
                metrics_csv=metrics_csv_path,
                figures_dir=figures_dir,
            )

            log("INFO", format_metrics("train", train_metrics))
            log("INFO", format_metrics("val", val_metrics))
            log("INFO", f"grad info: {json.dumps(jsonable(grad_info), ensure_ascii=False)}")
            log("INFO", f"epoch={epoch}, seconds={epoch_seconds}, lr_backbone={lr_backbone}, lr_head={lr_head}")

            save_checkpoint(
                checkpoints_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                metrics=row,
                args=args,
                best_val_iou=best_val_iou,
                best_epoch=best_epoch,
            )

            if int(args.save_every_n_epochs) > 0 and epoch % int(args.save_every_n_epochs) == 0:
                save_checkpoint(
                    checkpoints_dir / f"epoch_{epoch:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    metrics=row,
                    args=args,
                    best_val_iou=best_val_iou,
                    best_epoch=best_epoch,
                )

            if is_best:
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    metrics=row,
                    args=args,
                    best_val_iou=best_val_iou,
                    best_epoch=best_epoch,
                )

                log("OK", f"New best checkpoint saved at epoch {epoch}: val_iou_favela={best_val_iou:.4f}")

            clear_torch_memory()

        log("STEP", "Loading best checkpoint for threshold sweep and final test")
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

            validation_sweep_csv = threshold_dir / "validation_threshold_sweep.csv"
            write_csv(validation_sweep_csv, val_sweep_rows)

            save_threshold_sweep_curves(
                validation_sweep_csv=validation_sweep_csv,
                figures_dir=threshold_dir / "figures",
            )

            selected = select_best_threshold(val_sweep_rows, metric=str(args.selection_metric))
            selected_threshold = float(selected["threshold"])

            test_rows = evaluate_thresholds(
                model=model,
                loader=test_loader,
                device=device,
                thresholds=[selected_threshold],
                split_name="test",
                max_batches=args.max_test_batches,
            )

            test_selected = test_rows[0]
            write_csv(threshold_dir / "test_metrics_selected_threshold.csv", test_rows)

            log("OK", f"Selected threshold: {selected_threshold:.4f}")
            log("OK", f"Validation favela IoU at selected threshold: {selected['iou_favela']:.4f}")
            log("OK", f"Validation macro IoU at selected threshold: {selected['macro_iou']:.4f}")
            log("OK", f"Test favela IoU at selected threshold: {test_selected['iou_favela']:.4f}")
            log("OK", f"Test favela Dice at selected threshold: {test_selected['dice_favela']:.4f}")
            log("OK", f"Test macro IoU at selected threshold: {test_selected['macro_iou']:.4f}")
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

        final_summary = {
            "status": "completed",
            "run_dir": path_to_str(run_dir),
            "best_epoch": best_epoch,
            "best_val_iou_favela_threshold_0_5": best_val_iou,
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
        log("OK", f"Best val favela IoU at threshold 0.5: {best_val_iou:.4f}")
        log("OK", f"Selected threshold: {selected_threshold}")
        log("OK", f"Test favela IoU selected threshold: {test_selected.get('iou_favela')}")
        log("OK", f"Test favela Dice selected threshold: {test_selected.get('dice_favela')}")
        log("OK", f"Test no-favela IoU selected threshold: {test_selected.get('iou_no_favela')}")
        log("OK", f"Test macro IoU selected threshold: {test_selected.get('macro_iou')}")

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
        description="Train reBEN ResNet18 S1+S2 + UPerNet for favela segmentation."
    )

    parser.add_argument("--instance-root", required=True)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume training from a checkpoint. Use 'latest', 'best', or an explicit .pt path. "
            "When resuming, --epochs is the final target epoch, not the number of extra epochs."
        ),
    )

    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--s1-band-indices", default=None)
    parser.add_argument("--s2-band-indices", default=None)

    parser.add_argument("--reben-repo-id", default=DEFAULT_REBEN_REPO_ID)
    parser.add_argument("--reben-local-dir", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--min-loaded-backbone-keys", type=int, default=80)
    parser.add_argument("--allow-partial-pretrained", action="store_true")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)

    parser.add_argument("--out-indices", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--ppm-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument(
        "--freeze-backbone-mode",
        choices=["all", "layer4", "layer3_layer4", "none"],
        default="all",
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
        choices=["iou", "iou_favela", "dice", "dice_favela", "macro_iou", "macro_f1", "balanced_accuracy"],
        default="iou_favela",
    )

    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=0,
        help=(
            "If > 0, also save checkpoints/epoch_XXXX.pt every N epochs. "
            "latest.pt and best.pt are always saved."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())