#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
6_train_upernet_loro_dense_croma_224.py

Main objective
--------------
Train a UPerNet-style segmentation decoder using frozen dense CROMA features
for the S2 + SNAP-GRD VV/VH modality under a Leave-One-Region-Out split.

Input feature tensor:
    N x 768 x 28 x 28

Target label tensor:
    B x 1 x 224 x 224

The model upsamples dense CROMA features from 28x28 to 224x224 and predicts
a binary favela segmentation mask.

This script is designed to run a smoke test first.

Recommended smoke test
----------------------
python src/splitting_strategy_experiments/6_train_upernet_loro_dense_croma_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --heldout-region South `
  --epochs 2 `
  --batch-size 4 `
  --max-train-batches 10 `
  --max-val-batches 5 `
  --num-workers 0 `
  --overwrite

Recommended first longer run
----------------------------
python src/splitting_strategy_experiments/6_train_upernet_loro_dense_croma_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --heldout-region South `
  --epochs 20 `
  --batch-size 8 `
  --num-workers 0 `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
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


EXPECTED_FEATURE_SHAPE = (12699, 768, 28, 28)
REGIONS = ["Central-West", "North", "Northeast", "South", "Southeast"]


# ---------------------------------------------------------------------
# Logging and utilities
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


def slug_region(region: str) -> str:
    text = str(region).strip()
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(force_cpu: bool, device_index: int) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------

def default_dense_feature_path(instance_root: Path) -> Path:
    return (
        instance_root
        / "metadata"
        / "splitting_strategy_experiments"
        / "croma_dense_features_ps224_st112_cover"
        / "croma_dense_features_s2_s1_snap_vv_vh_joint_encodings_ps224_st112_cover_float16.npy"
    )


def default_alignment_dir(instance_root: Path) -> Path:
    return (
        instance_root
        / "metadata"
        / "splitting_strategy_experiments"
        / "dense_loro_alignment_validation_ps224_st112_cover"
    )


def default_training_root(instance_root: Path) -> Path:
    return (
        instance_root
        / "experiments"
        / "splitting_strategy_experiments"
        / "upernet_loro_dense_croma_ps224_st112_cover"
    )


def split_csv_path(alignment_dir: Path, heldout_region: str, split: str) -> Path:
    return alignment_dir / f"loro_fold_{slug_region(heldout_region)}_{split}_with_dense_index.csv"


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def resolve_path(path_value: Any, instance_root: Path) -> Path:
    raw = str(path_value).strip().replace("\\", "/")

    if raw == "":
        fail("Encountered empty path value.")

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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def read_label_window(
    label_path: Path,
    row_off: int,
    col_off: int,
    patch_size: int,
) -> np.ndarray:
    with rasterio.open(label_path) as src:
        window = Window(
            col_off=int(col_off),
            row_off=int(row_off),
            width=int(patch_size),
            height=int(patch_size),
        )

        arr = src.read(
            indexes=1,
            window=window,
            boundless=True,
            fill_value=0,
            out_shape=(patch_size, patch_size),
        ).astype(np.float32)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = (arr > 0.5).astype(np.float32)
    return arr[None, :, :]


class DenseCromaSegmentationDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        dense_feature_path: Path,
        instance_root: Path,
        patch_size: int = 224,
        feature_scale: float = 1.0,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.dense_feature_path = Path(dense_feature_path)
        self.instance_root = Path(instance_root)
        self.patch_size = int(patch_size)
        self.feature_scale = float(feature_scale)

        if not self.split_csv.exists():
            fail(f"Split CSV does not exist:\n{path_to_str(self.split_csv)}")

        if not self.dense_feature_path.exists():
            fail(f"Dense feature file does not exist:\n{path_to_str(self.dense_feature_path)}")

        self.df = pd.read_csv(self.split_csv)

        if self.df.empty:
            fail(f"Split CSV is empty:\n{path_to_str(self.split_csv)}")

        required_cols = ["dense_index", "patch_id", "label_path", "row_start", "col_start"]

        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            fail(
                f"Split CSV missing required columns: {missing}\n"
                f"Available columns: {list(self.df.columns)}"
            )

        self.df["dense_index"] = pd.to_numeric(self.df["dense_index"], errors="coerce").astype(int)
        self.df["row_start"] = pd.to_numeric(self.df["row_start"], errors="coerce").astype(int)
        self.df["col_start"] = pd.to_numeric(self.df["col_start"], errors="coerce").astype(int)

        self.features = np.load(self.dense_feature_path, mmap_mode="r")

        if self.features.ndim != 4:
            fail(f"Dense features must be N x C x H x W. Got {self.features.shape}")

        max_index = int(self.df["dense_index"].max())
        if max_index >= self.features.shape[0]:
            fail(
                f"Split references dense_index={max_index}, "
                f"but dense feature file has only {self.features.shape[0]} rows."
            )

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]

        dense_index = int(row["dense_index"])

        feature_np = np.asarray(self.features[dense_index], dtype=np.float32)
        if self.feature_scale != 1.0:
            feature_np = feature_np * self.feature_scale

        label_path = resolve_path(row["label_path"], self.instance_root)
        label_np = read_label_window(
            label_path=label_path,
            row_off=safe_int(row["row_start"]),
            col_off=safe_int(row["col_start"]),
            patch_size=self.patch_size,
        )

        sample = {
            "features": torch.from_numpy(feature_np),
            "mask": torch.from_numpy(label_np),
            "dense_index": torch.tensor(dense_index, dtype=torch.long),
            "patch_id": str(row["patch_id"]),
            "city": str(row["city"]) if "city" in row.index else "",
            "region": str(row["region"]) if "region" in row.index else "",
            "label_positive_pixels": float(row["label_positive_pixels"]) if "label_positive_pixels" in row.index else float(label_np.sum()),
        }

        return sample


# ---------------------------------------------------------------------
# Model: single-scale UPerNet-style decoder
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
                    ConvGNReLU(in_channels, ppm_channels, kernel_size=1),
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
    """
    UPerNet-style decoder for single-scale CROMA dense features.

    Input:
        B x 768 x 28 x 28

    Output:
        B x 1 x 224 x 224

    Since we currently have one CROMA dense feature scale, this is not a full
    multi-backbone-level UPerNet. It is a UPerNet-style decoder using:
      - input projection
      - pyramid pooling
      - progressive upsampling
      - binary segmentation head
    """

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
# Losses and metrics
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
        loss_value: float,
        threshold: float = 0.5,
    ) -> None:
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            targets = (targets >= 0.5).float()

            tp = (preds * targets).sum().item()
            fp = (preds * (1.0 - targets)).sum().item()
            fn = ((1.0 - preds) * targets).sum().item()
            tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += tn
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
            "loss": self.loss_sum / max(1, self.n_batches),
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "pred_pos_pct": 100.0 * self.pred_pos_pixels / max(eps, self.n_pixels),
            "gt_pos_pct": 100.0 * self.gt_pos_pixels / max(eps, self.n_pixels),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
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
    total_pixels = len(df) * int(patch_size) * int(patch_size)
    neg_pixels = total_pixels - pos_pixels

    if pos_pixels <= 0:
        warn("No positive pixels found in training CSV. Falling back to pos_weight=1.0.")
        return 1.0, {
            "method": "fallback",
            "reason": "no positive pixels",
            "pos_weight": 1.0,
        }

    raw_pos_weight = float(neg_pixels / pos_pixels)
    clipped = float(min(raw_pos_weight, max_pos_weight))

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
# Preview generation
# ---------------------------------------------------------------------

def save_prediction_previews(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    max_items: int = 4,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warn(f"matplotlib not available; skipping prediction previews: {exc}")
        return

    ensure_dir(output_dir)

    model.eval()
    saved = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device=device, dtype=torch.float32, non_blocking=True)
            masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

            logits = model(features)
            probs = torch.sigmoid(logits)

            b = features.shape[0]

            for i in range(b):
                if saved >= max_items:
                    return

                gt = masks[i, 0].detach().cpu().numpy()
                prob = probs[i, 0].detach().cpu().numpy()
                pred = (prob >= 0.5).astype(np.float32)

                patch_id = batch["patch_id"][i] if isinstance(batch["patch_id"], list) else str(saved)

                fig, axes = plt.subplots(1, 3, figsize=(10, 3))
                axes[0].imshow(gt, vmin=0, vmax=1)
                axes[0].set_title("Ground truth")
                axes[0].axis("off")

                axes[1].imshow(prob, vmin=0, vmax=1)
                axes[1].set_title("Probability")
                axes[1].axis("off")

                axes[2].imshow(pred, vmin=0, vmax=1)
                axes[2].set_title("Prediction")
                axes[2].axis("off")

                fig.suptitle(f"Epoch {epoch} | {patch_id}", fontsize=9)
                fig.tight_layout()

                out_path = output_dir / f"epoch{epoch:03d}_sample{saved:02d}.png"
                fig.savefig(out_path, dpi=150)
                plt.close(fig)

                saved += 1


# ---------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------

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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    max_batches: Optional[int],
) -> Dict[str, float]:
    model.train()

    acc = MetricAccumulator()

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=f"train epoch {epoch}", unit="batch")

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        features = batch["features"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(features)
                loss = criterion(logits, masks)

            if scaler is None:
                fail("AMP requested but scaler is None.")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        loss_value = float(loss.detach().cpu().item())

        if not math.isfinite(loss_value):
            fail(f"Non-finite training loss at batch {batch_idx}: {loss_value}")

        acc.update(logits.detach(), masks.detach(), loss_value=loss_value, threshold=0.5)

    return acc.compute()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    max_batches: Optional[int],
    split_name: str = "val",
) -> Dict[str, float]:
    model.eval()

    acc = MetricAccumulator()

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=f"{split_name} epoch {epoch}", unit="batch")

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        features = batch["features"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        logits = model(features)
        loss = criterion(logits, masks)
        loss_value = float(loss.detach().cpu().item())

        if not math.isfinite(loss_value):
            fail(f"Non-finite {split_name} loss at batch {batch_idx}: {loss_value}")

        acc.update(logits.detach(), masks.detach(), loss_value=loss_value, threshold=0.5)

    return acc.compute()


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
# Main
# ---------------------------------------------------------------------

def run_training(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))

    instance_root = Path(args.instance_root)

    heldout_region = str(args.heldout_region)
    if heldout_region not in REGIONS:
        fail(f"Unsupported heldout region: {heldout_region}. Expected one of {REGIONS}")

    dense_feature_path = Path(args.dense_feature_path) if args.dense_feature_path else default_dense_feature_path(instance_root)
    alignment_dir = Path(args.alignment_dir) if args.alignment_dir else default_alignment_dir(instance_root)

    train_csv = split_csv_path(alignment_dir, heldout_region, "train")
    val_csv = split_csv_path(alignment_dir, heldout_region, "val")
    test_csv = split_csv_path(alignment_dir, heldout_region, "test")

    training_root = Path(args.output_root) if args.output_root else default_training_root(instance_root)

    run_name = args.run_name
    if run_name is None:
        run_name = (
            f"heldout_{slug_region(heldout_region)}_"
            f"epochs{args.epochs}_bs{args.batch_size}_"
            f"dc{args.decoder_channels}"
        )
        if args.max_train_batches is not None or args.max_val_batches is not None:
            run_name += "_smoke"

    run_dir = training_root / run_name
    ensure_run_dir(run_dir, overwrite=bool(args.overwrite))

    checkpoints_dir = run_dir / "checkpoints"
    previews_dir = run_dir / "previews"
    ensure_dir(checkpoints_dir)
    ensure_dir(previews_dir)

    log("=" * 100, "")
    log("STEP", "UPerNet-style dense CROMA LORO training")
    log("INFO", f"Instance root:      {path_to_str(instance_root)}")
    log("INFO", f"Held-out region:    {heldout_region}")
    log("INFO", f"Dense feature path: {path_to_str(dense_feature_path)}")
    log("INFO", f"Alignment dir:      {path_to_str(alignment_dir)}")
    log("INFO", f"Train CSV:          {path_to_str(train_csv)}")
    log("INFO", f"Val CSV:            {path_to_str(val_csv)}")
    log("INFO", f"Test CSV:           {path_to_str(test_csv)}")
    log("INFO", f"Run dir:            {path_to_str(run_dir)}")
    log("=" * 100, "")

    if not dense_feature_path.exists():
        fail(f"Dense feature file not found:\n{path_to_str(dense_feature_path)}")

    dense_shape = tuple(np.load(dense_feature_path, mmap_mode="r").shape)
    log("INFO", f"Dense feature shape: {dense_shape}")

    if len(dense_shape) != 4:
        fail(f"Dense feature file must be 4D N x C x H x W. Got {dense_shape}")

    if dense_shape[1:] != (768, 28, 28):
        warn(f"Expected dense feature trailing shape (768, 28, 28), got {dense_shape[1:]}.")

    device = choose_device(bool(args.force_cpu), int(args.device_index))
    log("INFO", f"Selected device: {device}")

    pin_memory = bool(args.pin_memory) and device.type == "cuda"

    train_dataset = DenseCromaSegmentationDataset(
        split_csv=train_csv,
        dense_feature_path=dense_feature_path,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        feature_scale=float(args.feature_scale),
    )

    val_dataset = DenseCromaSegmentationDataset(
        split_csv=val_csv,
        dense_feature_path=dense_feature_path,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        feature_scale=float(args.feature_scale),
    )

    test_dataset = DenseCromaSegmentationDataset(
        split_csv=test_csv,
        dense_feature_path=dense_feature_path,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        feature_scale=float(args.feature_scale),
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

    model = SingleScaleUPerNetDecoder(
        in_channels=int(dense_shape[1]),
        decoder_channels=int(args.decoder_channels),
        ppm_channels=int(args.ppm_channels),
        dropout=float(args.dropout),
        out_channels=1,
    ).to(device)

    criterion = BCEDiceLoss(
        pos_weight=float(pos_weight),
        bce_weight=float(args.bce_weight),
        dice_weight=float(args.dice_weight),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

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
        "dense_feature_shape": dense_shape,
        "train_csv": path_to_str(train_csv),
        "val_csv": path_to_str(val_csv),
        "test_csv": path_to_str(test_csv),
        "train_patches": len(train_dataset),
        "val_patches": len(val_dataset),
        "test_patches": len(test_dataset),
        "pos_weight_info": pos_weight_info,
        "model": {
            "name": "SingleScaleUPerNetDecoder",
            "in_channels": int(dense_shape[1]),
            "decoder_channels": int(args.decoder_channels),
            "ppm_channels": int(args.ppm_channels),
            "dropout": float(args.dropout),
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

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                use_amp=use_amp,
                scaler=scaler,
                max_batches=args.max_train_batches,
            )

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                max_batches=args.max_val_batches,
                split_name="val",
            )

            scheduler.step(val_metrics["iou"])

            epoch_seconds = round(time.time() - epoch_started, 3)
            lr_current = float(optimizer.param_groups[0]["lr"])

            row = {
                "epoch": epoch,
                "seconds": epoch_seconds,
                "lr": lr_current,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            metrics_rows.append(row)

            write_csv(run_dir / "metrics.csv", metrics_rows)

            log("INFO", format_metrics("train", train_metrics))
            log("INFO", format_metrics("val", val_metrics))
            log("INFO", f"epoch={epoch}, seconds={epoch_seconds}, lr={lr_current:.8f}")

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

            if int(args.preview_every) > 0 and epoch % int(args.preview_every) == 0:
                save_prediction_previews(
                    model=model,
                    loader=val_loader,
                    device=device,
                    output_dir=previews_dir / f"epoch_{epoch:03d}",
                    epoch=epoch,
                    max_items=int(args.preview_items),
                )

        # Evaluate test once after training using latest model state.
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            epoch=int(args.epochs),
            max_batches=args.max_test_batches,
            split_name="test",
        )

        log("INFO", format_metrics("test", test_metrics))

        final_summary = {
            "status": "completed",
            "heldout_region": heldout_region,
            "run_dir": path_to_str(run_dir),
            "best_epoch": best_epoch,
            "best_val_iou": best_val_iou,
            "final_test_metrics_threshold_0_5": test_metrics,
            "epochs": int(args.epochs),
            "elapsed_seconds": round(time.time() - started, 3),
            "created_utc": now_utc(),
            "config": config,
        }

        write_json(run_dir / "final_summary.json", final_summary)

        save_prediction_previews(
            model=model,
            loader=test_loader,
            device=device,
            output_dir=previews_dir / "test_final",
            epoch=int(args.epochs),
            max_items=int(args.preview_items),
        )

        log("OK", "Training completed.")
        log("OK", f"Run dir: {path_to_str(run_dir)}")
        log("OK", f"Best epoch: {best_epoch}")
        log("OK", f"Best val IoU: {best_val_iou:.4f}")
        log("OK", f"Test IoU at threshold 0.5: {test_metrics['iou']:.4f}")

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
        description="Train UPerNet-style decoder on dense CROMA features using LORO splits."
    )

    parser.add_argument(
        "--instance-root",
        required=True,
        help="Dataset instance root.",
    )
    parser.add_argument(
        "--heldout-region",
        required=True,
        choices=REGIONS,
        help="Held-out LORO test region.",
    )
    parser.add_argument(
        "--dense-feature-path",
        default=None,
        help="Optional explicit dense feature .npy path.",
    )
    parser.add_argument(
        "--alignment-dir",
        default=None,
        help="Optional explicit dense/LORO alignment validation directory.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root for training runs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run name.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Label patch size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 on Windows for safety.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Optional cap for test evaluation.",
    )

    parser.add_argument(
        "--decoder-channels",
        type=int,
        default=256,
        help="Decoder base channels.",
    )
    parser.add_argument(
        "--ppm-channels",
        type=int,
        default=64,
        help="Pyramid pooling branch channels.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Decoder dropout.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="ReduceLROnPlateau patience.",
    )

    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.5,
        help="BCE component weight.",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.5,
        help="Dice component weight.",
    )
    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=50.0,
        help="Clip BCE positive class weight.",
    )

    parser.add_argument(
        "--feature-scale",
        type=float,
        default=1.0,
        help="Optional scale applied to dense CROMA features.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision on CUDA. Default enabled.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable automatic mixed precision.",
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
        help="Force CPU.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Use DataLoader pin_memory on CUDA. Default enabled.",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pin_memory.",
    )

    parser.add_argument(
        "--preview-every",
        type=int,
        default=1,
        help="Save prediction previews every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--preview-items",
        type=int,
        default=4,
        help="Number of preview samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite/reuse non-empty run directory.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())