#!/usr/bin/env python3
"""
Train a small U-Net baseline for Brazil favela segmentation.

Purpose
-------
This script trains a simple U-Net segmentation baseline using the PyTorch GeoTIFF
dataset loader from:

    16_build_pytorch_geotiff_dataset.py

It reads patches directly from GeoTIFFs using a patch filter-set CSV and applies
training-only normalization statistics.

It does NOT export H5 files.

Default setup
-------------
Patch list:
    <output_root>/metadata/patch_filter_sets_train_covered_region_test_ps512_st512_cover/
        filter_set_F02_quality_pass.csv

Normalization:
    <output_root>/metadata/
        normalization_stats_train_covered_region_test_ps512_st512_cover_F02_quality_pass.json

Default first baseline:
    modality: s2
    model: small U-Net
    loss: BCEWithLogitsLoss + Dice loss
    metrics: IoU, Dice, precision, recall

Recommended smoke test on CPU
-----------------------------
    python3 src/favela_postprocessing/18_train_unet_baseline.py \
        --config configs/default.yaml \
        --modality s2 \
        --epochs 2 \
        --batch-size 1 \
        --base-channels 16 \
        --max-train-patches 20 \
        --max-val-patches 10

Full S2-only baseline
---------------------
    python3 src/favela_postprocessing/18_train_unet_baseline.py \
        --config configs/default.yaml \
        --modality s2 \
        --epochs 20 \
        --batch-size 2 \
        --base-channels 32

Later modality comparisons
--------------------------
    --modality s2
    --modality s1
    --modality s2_s1
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SCRIPT_NAME = "18_train_unet_baseline.py"

DEFAULT_SPLIT_STRATEGY = "train_covered_region_test"
DEFAULT_PATCH_SIZE = 512
DEFAULT_STRIDE = 512
DEFAULT_EDGE_MODE = "cover"
DEFAULT_FILTER_SET_ID = "F02_quality_pass"


S2_BAND_SET_CHANNELS = {
    "all": 12,
    "rgb": 3,
    "rgb_nir": 4,
}

S1_CHANNEL_SET_CHANNELS = {
    "all": 3,
    "vv_vh": 2,
    "vv": 1,
    "vh": 1,
    "vvdiff": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small U-Net baseline for favela segmentation."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--filter-set",
        type=Path,
        default=None,
        help="Explicit filter-set CSV path. If omitted, uses default lookup.",
    )
    parser.add_argument(
        "--normalization-json",
        type=Path,
        default=None,
        help="Explicit normalization JSON path. If omitted, uses default lookup.",
    )
    parser.add_argument(
        "--filter-set-id",
        type=str,
        default=DEFAULT_FILTER_SET_ID,
        help=f"Filter set ID. Default: {DEFAULT_FILTER_SET_ID}",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default=DEFAULT_SPLIT_STRATEGY,
        help=f"Split strategy suffix. Default: {DEFAULT_SPLIT_STRATEGY}",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Patch size suffix. Default: {DEFAULT_PATCH_SIZE}",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Stride suffix. Default: {DEFAULT_STRIDE}",
    )
    parser.add_argument(
        "--edge-mode",
        type=str,
        default=DEFAULT_EDGE_MODE,
        help=f"Edge mode suffix. Default: {DEFAULT_EDGE_MODE}",
    )

    parser.add_argument(
        "--modality",
        type=str,
        default="s2",
        choices=["s2", "s1", "s2_s1"],
        help="Input modality. Default: s2",
    )
    parser.add_argument(
        "--s2-band-set",
        type=str,
        default="all",
        choices=["all", "rgb", "rgb_nir"],
        help="S2 band subset. Default: all",
    )
    parser.add_argument(
        "--s1-channel-set",
        type=str,
        default="all",
        choices=["all", "vv_vh", "vv", "vh", "vvdiff"],
        help="S1 channel subset. Default: all",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="standard",
        choices=["none", "standard", "clip_standard"],
        help="Input normalization mode. Default: standard",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs. Default: 2",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size. Default: 1",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=16,
        help="Base number of U-Net channels. Default: 16",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate. Default: 1e-4",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="AdamW weight decay. Default: 1e-5",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
        help="Weight of Dice loss in BCE + Dice. Default: 1.0",
    )
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=1.0,
        help="Weight of BCE loss in BCE + Dice. Default: 1.0",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Optional positive class weight for BCEWithLogitsLoss.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary metrics. Default: 0.5",
    )
    parser.add_argument(
        "--max-train-patches",
        type=int,
        default=None,
        help="Optional limit for training patches. Useful for smoke tests.",
    )
    parser.add_argument(
        "--max-val-patches",
        type=int,
        default=None,
        help="Optional limit for validation patches. Useful for smoke tests.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 for debugging. Default: 0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device. Default: auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Experiment output directory. If omitted, one is created automatically.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save a checkpoint at every epoch, not only best/latest.",
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


def default_suffix(
    split_strategy: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
) -> str:
    return f"{split_strategy}_ps{patch_size}_st{stride}_{edge_mode}"


def default_filter_set_path(
    output_root: Path,
    split_strategy: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
    filter_set_id: str,
) -> Path:
    suffix = default_suffix(split_strategy, patch_size, stride, edge_mode)

    return (
        output_root
        / "metadata"
        / f"patch_filter_sets_{suffix}"
        / f"filter_set_{filter_set_id}.csv"
    )


def default_normalization_json_path(
    output_root: Path,
    split_strategy: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
    filter_set_id: str,
) -> Path:
    suffix = default_suffix(split_strategy, patch_size, stride, edge_mode)

    return output_root / "metadata" / f"normalization_stats_{suffix}_{filter_set_id}.json"


def make_experiment_dir(
    output_root: Path,
    modality: str,
    filter_set_id: str,
    split_strategy: str,
    patch_size: int,
    base_channels: int,
    output_dir: Optional[Path],
) -> Path:
    if output_dir is not None:
        ensure_dir(output_dir)
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    name = (
        f"unet_{modality}_{filter_set_id}_"
        f"{split_strategy}_ps{patch_size}_bc{base_channels}_{timestamp}"
    )

    path = output_root / "experiments" / "unet_baseline" / name
    ensure_dir(path)

    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "You requested --device cuda, but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_dataset_module():
    """
    Load 16_build_pytorch_geotiff_dataset.py despite the leading numeric filename.
    """
    current_dir = Path(__file__).resolve().parent
    dataset_script = current_dir / "16_build_pytorch_geotiff_dataset.py"

    if not dataset_script.exists():
        raise FileNotFoundError(f"Dataset script not found: {dataset_script}")

    spec = importlib.util.spec_from_file_location(
        "favela_pytorch_geotiff_dataset",
        dataset_script,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for: {dataset_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["favela_pytorch_geotiff_dataset"] = module
    spec.loader.exec_module(module)

    return module


def infer_input_channels(
    modality: str,
    s2_band_set: str,
    s1_channel_set: str,
) -> int:
    channels = 0

    if modality in {"s2", "s2_s1"}:
        channels += S2_BAND_SET_CHANNELS[s2_band_set]

    if modality in {"s1", "s2_s1"}:
        channels += S1_CHANNEL_SET_CHANNELS[s1_channel_set]

    return channels


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    """
    Simple U-Net for binary segmentation.

    Input:
        [B, C, H, W]

    Output:
        logits [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c4, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c3, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c2, c1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)

        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self._match_and_concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_and_concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_and_concat(d1, e1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

    @staticmethod
    def _match_and_concat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Match spatial size defensively, then concatenate.
        """
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(
                up,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return torch.cat([skip, up], dim=1)


def batch_to_input(
    batch: Dict[str, Any],
    modality: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = []

    if modality in {"s2", "s2_s1"}:
        tensors.append(batch["s2"].float())

    if modality in {"s1", "s2_s1"}:
        tensors.append(batch["s1"].float())

    x = torch.cat(tensors, dim=1).to(device, non_blocking=True)
    y = batch["label"].float().to(device, non_blocking=True)
    valid_mask = batch["valid_mask"].float().to(device, non_blocking=True)

    return x, y, valid_mask


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )

    loss = loss * valid_mask

    denom = valid_mask.sum().clamp_min(1.0)

    return loss.sum() / denom


def masked_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)

    probs = probs * valid_mask
    targets = targets * valid_mask

    intersection = (probs * targets).sum()
    denominator = probs.sum() + targets.sum()

    dice = (2.0 * intersection + eps) / (denominator + eps)

    return 1.0 - dice


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
    pos_weight: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce = masked_bce_with_logits(
        logits=logits,
        targets=targets,
        valid_mask=valid_mask,
        pos_weight=pos_weight,
    )

    dice = masked_dice_loss(
        logits=logits,
        targets=targets,
        valid_mask=valid_mask,
    )

    total = bce_weight * bce + dice_weight * dice

    return total, {
        "bce_loss": float(bce.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


class MetricAccumulator:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.bce_loss_sum = 0.0
        self.dice_loss_sum = 0.0
        self.batch_count = 0

        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

        self.valid_pixels = 0.0
        self.positive_pixels = 0.0
        self.pred_positive_pixels = 0.0

    def update_loss(self, loss_parts: Dict[str, float]) -> None:
        self.loss_sum += loss_parts["total_loss"]
        self.bce_loss_sum += loss_parts["bce_loss"]
        self.dice_loss_sum += loss_parts["dice_loss"]
        self.batch_count += 1

    def update_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = probs >= self.threshold
            targets_bool = targets >= 0.5
            valid_bool = valid_mask >= 0.5

            tp = (preds & targets_bool & valid_bool).sum().item()
            fp = (preds & ~targets_bool & valid_bool).sum().item()
            fn = (~preds & targets_bool & valid_bool).sum().item()
            tn = (~preds & ~targets_bool & valid_bool).sum().item()

            self.tp += float(tp)
            self.fp += float(fp)
            self.fn += float(fn)
            self.tn += float(tn)

            self.valid_pixels += float(valid_bool.sum().item())
            self.positive_pixels += float((targets_bool & valid_bool).sum().item())
            self.pred_positive_pixels += float((preds & valid_bool).sum().item())

    def compute(self) -> Dict[str, float]:
        eps = 1e-6

        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)

        positive_pixel_percent = 100.0 * self.positive_pixels / max(self.valid_pixels, eps)
        pred_positive_pixel_percent = 100.0 * self.pred_positive_pixels / max(self.valid_pixels, eps)

        return {
            "loss": self.loss_sum / max(self.batch_count, 1),
            "bce_loss": self.bce_loss_sum / max(self.batch_count, 1),
            "dice_loss": self.dice_loss_sum / max(self.batch_count, 1),
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "valid_pixels": self.valid_pixels,
            "positive_pixels": self.positive_pixels,
            "pred_positive_pixels": self.pred_positive_pixels,
            "positive_pixel_percent": positive_pixel_percent,
            "pred_positive_pixel_percent": pred_positive_pixel_percent,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality: str,
    bce_weight: float,
    dice_weight: float,
    pos_weight: Optional[torch.Tensor],
    threshold: float,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    acc = MetricAccumulator(threshold=threshold)

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)

    for batch in pbar:
        x, y, valid_mask = batch_to_input(batch, modality, device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        loss, loss_parts = combined_loss(
            logits=logits,
            targets=y,
            valid_mask=valid_mask,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
        )

        loss.backward()
        optimizer.step()

        acc.update_loss(loss_parts)
        acc.update_metrics(logits.detach(), y, valid_mask)

        metrics = acc.compute()
        pbar.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            iou=f"{metrics['iou']:.4f}",
            dice=f"{metrics['dice']:.4f}",
        )

    return acc.compute()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    bce_weight: float,
    dice_weight: float,
    pos_weight: Optional[torch.Tensor],
    threshold: float,
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    acc = MetricAccumulator(threshold=threshold)

    pbar = tqdm(loader, desc=f"Val epoch {epoch}", leave=False)

    for batch in pbar:
        x, y, valid_mask = batch_to_input(batch, modality, device)

        logits = model(x)

        _, loss_parts = combined_loss(
            logits=logits,
            targets=y,
            valid_mask=valid_mask,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
        )

        acc.update_loss(loss_parts)
        acc.update_metrics(logits, y, valid_mask)

        metrics = acc.compute()
        pbar.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            iou=f"{metrics['iou']:.4f}",
            dice=f"{metrics['dice']:.4f}",
        )

    return acc.compute()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_iou: float,
    args_dict: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_iou": best_val_iou,
            "args": args_dict,
            "metrics": metrics,
        },
        path,
    )


def append_metrics_csv(
    path: Path,
    row: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)

    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def make_json_serializable(value: Any) -> Any:
    """
    Convert objects such as pathlib.Path, NumPy scalars, and torch/device objects
    into JSON-serializable Python types.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.device):
        return str(value)

    return value


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)

    serializable_data = make_json_serializable(data)

    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    output_root = Path(str(cfg["output_root"]))

    filter_set_path = (
        args.filter_set
        if args.filter_set is not None
        else default_filter_set_path(
            output_root=output_root,
            split_strategy=args.split_strategy,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            filter_set_id=args.filter_set_id,
        )
    )

    normalization_json_path = (
        args.normalization_json
        if args.normalization_json is not None
        else default_normalization_json_path(
            output_root=output_root,
            split_strategy=args.split_strategy,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            filter_set_id=args.filter_set_id,
        )
    )

    if args.normalization == "none":
        normalization_json_path = None

    experiment_dir = make_experiment_dir(
        output_root=output_root,
        modality=args.modality,
        filter_set_id=args.filter_set_id,
        split_strategy=args.split_strategy,
        patch_size=args.patch_size,
        base_channels=args.base_channels,
        output_dir=args.output_dir,
    )

    checkpoints_dir = experiment_dir / "checkpoints"
    ensure_dir(checkpoints_dir)

    metrics_csv = experiment_dir / "metrics.csv"
    config_json = experiment_dir / "config.json"
    final_metrics_json = experiment_dir / "final_metrics.json"

    device = resolve_device(args.device)

    dataset_module = load_dataset_module()
    DatasetClass = dataset_module.BrazilFavelaGeoTiffDataset

    print("[INFO] Train U-Net baseline")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Experiment dir: {experiment_dir}")
    print(f"[INFO] Filter set: {filter_set_path}")
    print(f"[INFO] Normalization JSON: {normalization_json_path}")
    print(f"[INFO] Modality: {args.modality}")
    print(f"[INFO] S2 band set: {args.s2_band_set}")
    print(f"[INFO] S1 channel set: {args.s1_channel_set}")
    print(f"[INFO] Normalization: {args.normalization}")
    print(f"[INFO] Epochs: {args.epochs}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

    train_dataset = DatasetClass(
        patch_csv=filter_set_path,
        split="train",
        modality=args.modality,
        normalization_json=normalization_json_path,
        normalization=args.normalization,
        s2_band_set=args.s2_band_set,
        s1_channel_set=args.s1_channel_set,
        max_patches=args.max_train_patches,
        treat_all_zero_s2_as_nodata=True,
    )

    val_dataset = DatasetClass(
        patch_csv=filter_set_path,
        split="val",
        modality=args.modality,
        normalization_json=normalization_json_path,
        normalization=args.normalization,
        s2_band_set=args.s2_band_set,
        s1_channel_set=args.s1_channel_set,
        max_patches=args.max_val_patches,
        treat_all_zero_s2_as_nodata=True,
    )

    print(f"[INFO] Train patches: {len(train_dataset)}")
    print(f"[INFO] Val patches: {len(val_dataset)}")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    in_channels = infer_input_channels(
        modality=args.modality,
        s2_band_set=args.s2_band_set,
        s1_channel_set=args.s1_channel_set,
    )

    model = SmallUNet(
        in_channels=in_channels,
        base_channels=args.base_channels,
        out_channels=1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    pos_weight_tensor: Optional[torch.Tensor]

    if args.pos_weight is not None:
        pos_weight_tensor = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
        print(f"[INFO] BCE pos_weight: {args.pos_weight}")
    else:
        pos_weight_tensor = None
        print("[INFO] BCE pos_weight: None")

    n_params = count_parameters(model)
    print(f"[INFO] Model input channels: {in_channels}")
    print(f"[INFO] Trainable parameters: {n_params:,}")

    args_dict = vars(args).copy()
    args_dict.update(
        {
            "filter_set_path": str(filter_set_path),
            "normalization_json_path": str(normalization_json_path) if normalization_json_path else None,
            "experiment_dir": str(experiment_dir),
            "device_resolved": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "input_channels": in_channels,
            "train_patches": len(train_dataset),
            "val_patches": len(val_dataset),
            "trainable_parameters": n_params,
        }
    )

    # Convert pathlib.Path, NumPy scalars, and torch/device objects to plain
    # JSON-safe Python types before saving config and checkpoint metadata.
    args_dict = make_json_serializable(args_dict)

    save_json(config_json, args_dict)

    best_val_iou = -1.0
    best_epoch = -1

    all_epoch_metrics: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            modality=args.modality,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            pos_weight=pos_weight_tensor,
            threshold=args.threshold,
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            modality=args.modality,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            pos_weight=pos_weight_tensor,
            threshold=args.threshold,
            epoch=epoch,
        )

        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_bce_loss": train_metrics["bce_loss"],
            "train_dice_loss": train_metrics["dice_loss"],
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_accuracy": train_metrics["accuracy"],
            "train_positive_pixel_percent": train_metrics["positive_pixel_percent"],
            "train_pred_positive_pixel_percent": train_metrics["pred_positive_pixel_percent"],
            "val_loss": val_metrics["loss"],
            "val_bce_loss": val_metrics["bce_loss"],
            "val_dice_loss": val_metrics["dice_loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],
            "val_positive_pixel_percent": val_metrics["positive_pixel_percent"],
            "val_pred_positive_pixel_percent": val_metrics["pred_positive_pixel_percent"],
        }

        all_epoch_metrics.append(row)
        append_metrics_csv(metrics_csv, row)

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} "
            f"train_iou={row['train_iou']:.4f} "
            f"train_dice={row['train_dice']:.4f} | "
            f"val_loss={row['val_loss']:.4f} "
            f"val_iou={row['val_iou']:.4f} "
            f"val_dice={row['val_dice']:.4f} "
            f"val_precision={row['val_precision']:.4f} "
            f"val_recall={row['val_recall']:.4f}"
        )

        latest_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }

        save_checkpoint(
            path=checkpoints_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_iou=best_val_iou,
            args_dict=args_dict,
            metrics=latest_metrics,
        )

        if args.save_every_epoch:
            save_checkpoint(
                path=checkpoints_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_iou=best_val_iou,
                args_dict=args_dict,
                metrics=latest_metrics,
            )

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch

            save_checkpoint(
                path=checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_iou=best_val_iou,
                args_dict=args_dict,
                metrics=latest_metrics,
            )

    final_output = {
        "script": SCRIPT_NAME,
        "experiment_dir": str(experiment_dir),
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "epochs": all_epoch_metrics,
        "config": args_dict,
    }

    save_json(final_metrics_json, final_output)

    print("[INFO] Training complete.")
    print(f"[INFO] Best epoch: {best_epoch}")
    print(f"[INFO] Best val IoU: {best_val_iou:.6f}")
    print(f"[INFO] Metrics CSV: {metrics_csv}")
    print(f"[INFO] Best checkpoint: {checkpoints_dir / 'best.pt'}")
    print(f"[INFO] Latest checkpoint: {checkpoints_dir / 'latest.pt'}")
    print(f"[INFO] Final metrics JSON: {final_metrics_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())