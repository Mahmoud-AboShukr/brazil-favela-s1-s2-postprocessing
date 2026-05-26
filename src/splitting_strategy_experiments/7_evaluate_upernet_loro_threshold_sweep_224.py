#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
7_evaluate_upernet_loro_threshold_sweep_224.py

Main objective
--------------
Evaluate the best checkpoint from the dense-CROMA UPerNet-style LORO training
run using a validation threshold sweep.

Why this script exists
----------------------
Script 6 trains the model and saves best.pt/latest.pt, but the final test metric
was computed at a fixed threshold of 0.5 and from the final in-memory epoch.

For imbalanced binary segmentation, threshold 0.5 is often not optimal.
This script reloads checkpoints/best.pt, sweeps thresholds on validation only,
selects the best validation-IoU threshold, and evaluates the test set once using
that selected threshold.

Recommended command
-------------------
python src/splitting_strategy_experiments/7_evaluate_upernet_loro_threshold_sweep_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --heldout-region South `
  --run-dir "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired/experiments/splitting_strategy_experiments/upernet_loro_dense_croma_ps224_st112_cover/heldout_South_epochs20_bs8_dc256_full" `
  --batch-size 32 `
  --num-workers 0 `
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


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)

    existing = (
        list(path.glob("*.csv"))
        + list(path.glob("*.json"))
        + list(path.glob("*.md"))
        + list(path.glob("*.png"))
    )

    if existing and not overwrite:
        fail(
            "Output directory already contains files:\n"
            f"{path_to_str(path)}\n\n"
            "Use --overwrite to replace evaluation outputs."
        )


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

        return {
            "features": torch.from_numpy(feature_np),
            "mask": torch.from_numpy(label_np),
            "dense_index": torch.tensor(dense_index, dtype=torch.long),
            "patch_id": str(row["patch_id"]),
            "city": str(row["city"]) if "city" in row.index else "",
            "region": str(row["region"]) if "region" in row.index else "",
            "label_positive_pixels": float(row["label_positive_pixels"]) if "label_positive_pixels" in row.index else float(label_np.sum()),
        }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )


# ---------------------------------------------------------------------
# Model definition identical to script 6
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
# Metrics
# ---------------------------------------------------------------------

@dataclass
class BinaryMetricAccumulator:
    threshold: float
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0
    n_pixels: float = 0.0
    pred_pos_pixels: float = 0.0
    gt_pos_pixels: float = 0.0

    def update_from_probs(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        with torch.no_grad():
            preds = (probs >= float(self.threshold)).float()
            targets = (targets >= 0.5).float()

            tp = (preds * targets).sum().item()
            fp = (preds * (1.0 - targets)).sum().item()
            fn = ((1.0 - preds) * targets).sum().item()
            tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += tn
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


def make_thresholds(start: float, end: float, step: float) -> List[float]:
    values: List[float] = []
    x = float(start)

    while x <= float(end) + 1e-9:
        values.append(round(x, 6))
        x += float(step)

    return values


@torch.no_grad()
def evaluate_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: Sequence[float],
    split_name: str,
    max_batches: Optional[int] = None,
) -> List[Dict[str, float]]:
    model.eval()

    accumulators = [BinaryMetricAccumulator(threshold=t) for t in thresholds]

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=f"{split_name} threshold eval", unit="batch")

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        features = batch["features"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        logits = model(features)
        probs = torch.sigmoid(logits)

        for acc in accumulators:
            acc.update_from_probs(probs, targets)

    rows = [acc.compute() for acc in accumulators]
    return rows


def select_best_threshold(rows: List[Dict[str, float]], metric: str = "iou") -> Dict[str, float]:
    if not rows:
        fail("No threshold rows available.")

    if metric not in rows[0]:
        fail(f"Metric '{metric}' not found in threshold rows.")

    # Tie-breaker:
    # 1. highest selected metric
    # 2. highest Dice
    # 3. predicted positive percentage closest to GT positive percentage
    best = sorted(
        rows,
        key=lambda r: (
            float(r[metric]),
            float(r["dice"]),
            -abs(float(r["pred_pos_pct"]) - float(r["gt_pos_pct"])),
        ),
        reverse=True,
    )[0]

    return best


# ---------------------------------------------------------------------
# Checkpoint and previews
# ---------------------------------------------------------------------

def load_checkpoint(path: Path, map_location: torch.device) -> Dict[str, Any]:
    if not path.exists():
        fail(f"Checkpoint does not exist:\n{path_to_str(path)}")

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    fallback_decoder_channels: int,
    fallback_ppm_channels: int,
    fallback_dropout: float,
) -> nn.Module:
    ckpt_args = checkpoint.get("args", {}) or {}

    decoder_channels = int(ckpt_args.get("decoder_channels", fallback_decoder_channels))
    ppm_channels = int(ckpt_args.get("ppm_channels", fallback_ppm_channels))
    dropout = float(ckpt_args.get("dropout", fallback_dropout))

    model = SingleScaleUPerNetDecoder(
        in_channels=768,
        decoder_channels=decoder_channels,
        ppm_channels=ppm_channels,
        dropout=dropout,
        out_channels=1,
    )

    state = checkpoint.get("model_state_dict")
    if state is None:
        fail("Checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(state, strict=True)
    return model


def save_prediction_previews(
    model: nn.Module,
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


# ---------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------

def write_markdown_report(
    path: Path,
    payload: Dict[str, Any],
    val_rows: List[Dict[str, float]],
    test_selected: Dict[str, float],
) -> None:
    lines: List[str] = []

    lines.append("# UPerNet Dense-CROMA LORO Threshold Sweep Evaluation")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Reload the best checkpoint, select the decision threshold using validation IoU, "
        "and evaluate the held-out test region once using that validation-selected threshold."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    for key in [
        "instance_root",
        "heldout_region",
        "run_dir",
        "checkpoint_path",
        "dense_feature_path",
        "alignment_dir",
        "val_csv",
        "test_csv",
        "output_dir",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")

    lines.append("")
    lines.append("## Selected Threshold")
    lines.append("")
    selected = payload["selected_threshold"]
    lines.append(f"- Selection metric: `{payload['selection_metric']}`")
    lines.append(f"- Selected threshold: `{selected['threshold']}`")
    lines.append(f"- Validation IoU: `{selected['iou']:.6f}`")
    lines.append(f"- Validation Dice: `{selected['dice']:.6f}`")
    lines.append(f"- Validation precision: `{selected['precision']:.6f}`")
    lines.append(f"- Validation recall: `{selected['recall']:.6f}`")
    lines.append(f"- Validation predicted positive %: `{selected['pred_pos_pct']:.6f}`")
    lines.append(f"- Validation GT positive %: `{selected['gt_pos_pct']:.6f}`")
    lines.append("")
    lines.append("## Test Metrics at Selected Threshold")
    lines.append("")
    for key in [
        "threshold",
        "iou",
        "dice",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "pred_pos_pct",
        "gt_pos_pct",
    ]:
        value = test_selected[key]
        if isinstance(value, float):
            lines.append(f"- {key}: `{value:.6f}`")
        else:
            lines.append(f"- {key}: `{value}`")

    lines.append("")
    lines.append("## Validation Threshold Sweep")
    lines.append("")
    lines.append("| Threshold | IoU | Dice | Precision | Recall | Pred + % | GT + % |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    for row in val_rows:
        lines.append(
            f"| {row['threshold']:.2f} "
            f"| {row['iou']:.6f} "
            f"| {row['dice']:.6f} "
            f"| {row['precision']:.6f} "
            f"| {row['recall']:.6f} "
            f"| {row['pred_pos_pct']:.6f} "
            f"| {row['gt_pos_pct']:.6f} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This evaluation should be used instead of the fixed-threshold 0.5 result from the training script. "
        "The threshold is selected only on validation, then applied once to the held-out test region."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))

    instance_root = Path(args.instance_root)
    heldout_region = str(args.heldout_region)

    if heldout_region not in REGIONS:
        fail(f"Unsupported heldout region: {heldout_region}. Expected one of {REGIONS}")

    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        fail(f"Run directory does not exist:\n{path_to_str(run_dir)}")

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else run_dir / "checkpoints" / "best.pt"
    dense_feature_path = Path(args.dense_feature_path) if args.dense_feature_path else default_dense_feature_path(instance_root)
    alignment_dir = Path(args.alignment_dir) if args.alignment_dir else default_alignment_dir(instance_root)

    val_csv = Path(args.val_csv) if args.val_csv else split_csv_path(alignment_dir, heldout_region, "val")
    test_csv = Path(args.test_csv) if args.test_csv else split_csv_path(alignment_dir, heldout_region, "test")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "evaluation_threshold_sweep_best"

    ensure_output_dir(output_dir, overwrite=bool(args.overwrite))

    log("=" * 100, "")
    log("STEP", "UPerNet dense-CROMA threshold sweep evaluation")
    log("INFO", f"Instance root:      {path_to_str(instance_root)}")
    log("INFO", f"Held-out region:    {heldout_region}")
    log("INFO", f"Run dir:            {path_to_str(run_dir)}")
    log("INFO", f"Checkpoint:         {path_to_str(checkpoint_path)}")
    log("INFO", f"Dense feature path: {path_to_str(dense_feature_path)}")
    log("INFO", f"Alignment dir:      {path_to_str(alignment_dir)}")
    log("INFO", f"Val CSV:            {path_to_str(val_csv)}")
    log("INFO", f"Test CSV:           {path_to_str(test_csv)}")
    log("INFO", f"Output dir:         {path_to_str(output_dir)}")
    log("=" * 100, "")

    dense_shape = tuple(np.load(dense_feature_path, mmap_mode="r").shape)
    log("INFO", f"Dense feature shape: {dense_shape}")

    device = choose_device(bool(args.force_cpu), int(args.device_index))
    log("INFO", f"Selected device: {device}")

    pin_memory = bool(args.pin_memory) and device.type == "cuda"

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

    val_loader = make_loader(
        val_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )

    test_loader = make_loader(
        test_dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )

    log("INFO", f"Val patches:  {len(val_dataset):,}")
    log("INFO", f"Test patches: {len(test_dataset):,}")

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    checkpoint_epoch = checkpoint.get("epoch", None)
    checkpoint_metrics = checkpoint.get("metrics", {})

    log("INFO", f"Loaded checkpoint epoch: {checkpoint_epoch}")
    if checkpoint_metrics:
        log("INFO", f"Checkpoint stored val_iou: {checkpoint_metrics.get('val_iou', 'NA')}")

    model = build_model_from_checkpoint(
        checkpoint=checkpoint,
        fallback_decoder_channels=int(args.decoder_channels),
        fallback_ppm_channels=int(args.ppm_channels),
        fallback_dropout=float(args.dropout),
    ).to(device)

    model.eval()

    thresholds = make_thresholds(
        start=float(args.threshold_start),
        end=float(args.threshold_end),
        step=float(args.threshold_step),
    )

    log("STEP", f"Sweeping {len(thresholds)} thresholds on validation")

    val_rows = evaluate_thresholds(
        model=model,
        loader=val_loader,
        device=device,
        thresholds=thresholds,
        split_name="val",
        max_batches=args.max_val_batches,
    )

    val_csv_path = output_dir / "validation_threshold_sweep.csv"
    write_csv(val_csv_path, val_rows)

    selected = select_best_threshold(val_rows, metric=str(args.selection_metric))
    selected_threshold = float(selected["threshold"])

    log("OK", f"Selected threshold: {selected_threshold:.4f}")
    log("OK", f"Selected validation IoU: {selected['iou']:.6f}")
    log("OK", f"Selected validation Dice: {selected['dice']:.6f}")
    log("OK", f"Selected validation precision/recall: {selected['precision']:.6f}/{selected['recall']:.6f}")

    log("STEP", "Evaluating test using selected validation threshold")

    test_rows = evaluate_thresholds(
        model=model,
        loader=test_loader,
        device=device,
        thresholds=[selected_threshold],
        split_name="test",
        max_batches=args.max_test_batches,
    )

    test_selected = test_rows[0]

    test_csv_path = output_dir / "test_metrics_selected_threshold.csv"
    write_csv(test_csv_path, test_rows)

    preview_val_dir = output_dir / "previews_val_selected_threshold"
    preview_test_dir = output_dir / "previews_test_selected_threshold"

    if int(args.preview_items) > 0:
        save_prediction_previews(
            model=model,
            loader=val_loader,
            device=device,
            output_dir=preview_val_dir,
            threshold=selected_threshold,
            max_items=int(args.preview_items),
            title_prefix=f"VAL {heldout_region}",
        )

        save_prediction_previews(
            model=model,
            loader=test_loader,
            device=device,
            output_dir=preview_test_dir,
            threshold=selected_threshold,
            max_items=int(args.preview_items),
            title_prefix=f"TEST {heldout_region}",
        )

    summary = {
        "status": "completed",
        "created_utc": now_utc(),
        "instance_root": path_to_str(instance_root),
        "heldout_region": heldout_region,
        "run_dir": path_to_str(run_dir),
        "checkpoint_path": path_to_str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_metrics": checkpoint_metrics,
        "dense_feature_path": path_to_str(dense_feature_path),
        "dense_feature_shape": dense_shape,
        "alignment_dir": path_to_str(alignment_dir),
        "val_csv": path_to_str(val_csv),
        "test_csv": path_to_str(test_csv),
        "output_dir": path_to_str(output_dir),
        "batch_size": int(args.batch_size),
        "thresholds": thresholds,
        "selection_metric": str(args.selection_metric),
        "selected_threshold": selected,
        "test_metrics_selected_threshold": test_selected,
        "validation_threshold_sweep_csv": path_to_str(val_csv_path),
        "test_metrics_selected_threshold_csv": path_to_str(test_csv_path),
        "preview_val_dir": path_to_str(preview_val_dir),
        "preview_test_dir": path_to_str(preview_test_dir),
        "args": vars(args),
    }

    summary_json_path = output_dir / "threshold_sweep_summary.json"
    report_md_path = output_dir / "threshold_sweep_report.md"

    write_json(summary_json_path, summary)
    write_markdown_report(
        path=report_md_path,
        payload=summary,
        val_rows=val_rows,
        test_selected=test_selected,
    )

    log("=" * 100, "")
    log("OK", "Threshold sweep evaluation completed.")
    log("OK", f"Validation sweep CSV: {path_to_str(val_csv_path)}")
    log("OK", f"Test selected-threshold CSV: {path_to_str(test_csv_path)}")
    log("OK", f"Summary JSON: {path_to_str(summary_json_path)}")
    log("OK", f"Report MD: {path_to_str(report_md_path)}")
    log("OK", f"Selected threshold: {selected_threshold:.4f}")
    log("OK", f"Validation IoU at selected threshold: {selected['iou']:.6f}")
    log("OK", f"Test IoU at selected threshold: {test_selected['iou']:.6f}")
    log("OK", f"Test Dice at selected threshold: {test_selected['dice']:.6f}")
    log("=" * 100, "")

    clear_torch_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense-CROMA UPerNet checkpoint with validation threshold sweep."
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
        "--run-dir",
        required=True,
        help="Training run directory containing checkpoints/best.pt.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional explicit checkpoint path. Default: run-dir/checkpoints/best.pt",
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
        "--val-csv",
        default=None,
        help="Optional explicit validation CSV.",
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="Optional explicit test CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory.",
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
        default=32,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 on Windows.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional validation batch cap for debugging.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Optional test batch cap for debugging.",
    )

    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.05,
        help="Threshold sweep start.",
    )
    parser.add_argument(
        "--threshold-end",
        type=float,
        default=0.95,
        help="Threshold sweep end.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Threshold sweep step.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["iou", "dice", "balanced_accuracy"],
        default="iou",
        help="Metric used to select threshold on validation.",
    )

    parser.add_argument(
        "--decoder-channels",
        type=int,
        default=256,
        help="Fallback decoder channels if checkpoint args are missing.",
    )
    parser.add_argument(
        "--ppm-channels",
        type=int,
        default=64,
        help="Fallback PPM channels if checkpoint args are missing.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Fallback dropout if checkpoint args are missing.",
    )

    parser.add_argument(
        "--feature-scale",
        type=float,
        default=1.0,
        help="Optional feature scale.",
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
        help="Use pin_memory on CUDA. Default enabled.",
    )
    parser.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pin_memory.",
    )

    parser.add_argument(
        "--preview-items",
        type=int,
        default=8,
        help="Number of validation/test preview images to save. Use 0 to disable.",
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
        help="Overwrite existing evaluation output files.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_evaluation(parse_args())
    except Exception:
        traceback.print_exc()
        raise