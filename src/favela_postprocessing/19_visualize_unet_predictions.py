#!/usr/bin/env python3
"""
Visualize U-Net baseline predictions for Brazil favela segmentation.

Purpose
-------
This script loads a trained U-Net checkpoint from:

    18_train_unet_baseline.py

and generates visual QC figures showing:

    - Sentinel-2 RGB
    - Sentinel-2 false color
    - Sentinel-1 VV/VH/VV-minus-VH
    - ground-truth label
    - predicted probability map
    - binary prediction
    - TP/FP/FN/TN error map
    - prediction overlay on S2 RGB

It does NOT train a model.
It does NOT export H5 files.

Why this matters
----------------
Metrics alone are not enough at this stage. Prediction visualization helps us see
whether the model is learning meaningful favela-like structures or simply predicting
noise/background.

Example
-------
Use the best checkpoint from an experiment:

    python3 src/favela_postprocessing/19_visualize_unet_predictions.py \
        --config configs/default.yaml \
        --checkpoint /media/HALLOPEAU/T7/post_processing_dataset/experiments/unet_baseline/<experiment_name>/checkpoints/best.pt \
        --split val \
        --sample-mode mixed \
        --num-samples 12

If --checkpoint is omitted, the script tries to find the most recently modified
best.pt under:

    <output_root>/experiments/unet_baseline/*/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


SCRIPT_NAME = "19_visualize_unet_predictions.py"

DEFAULT_SPLIT_STRATEGY = "train_covered_region_test"
DEFAULT_PATCH_SIZE = 512
DEFAULT_STRIDE = 512
DEFAULT_EDGE_MODE = "cover"
DEFAULT_FILTER_SET_ID = "F02_quality_pass"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize U-Net baseline predictions."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to checkpoint .pt file. If omitted, the latest best.pt under "
            "<output_root>/experiments/unet_baseline is used."
        ),
    )
    parser.add_argument(
        "--filter-set",
        type=Path,
        default=None,
        help=(
            "Optional filter-set CSV path. If omitted, uses checkpoint config if available, "
            "otherwise default F02 path."
        ),
    )
    parser.add_argument(
        "--normalization-json",
        type=Path,
        default=None,
        help=(
            "Optional normalization JSON path. If omitted, uses checkpoint config if available, "
            "otherwise default normalization stats."
        ),
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
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test", "all"],
        help="Dataset split to visualize. Default: val",
    )
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="mixed",
        choices=["first", "random", "positive", "negative", "mixed", "high_positive"],
        help="How to select patches. Default: mixed",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="Number of samples to visualize. Default: 12",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Prediction threshold. If omitted, uses checkpoint config threshold or 0.5.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        choices=["s2", "s1", "s2_s1"],
        help="Override modality. If omitted, uses checkpoint config.",
    )
    parser.add_argument(
        "--s2-band-set",
        type=str,
        default=None,
        choices=["all", "rgb", "rgb_nir"],
        help="Override S2 band set. If omitted, uses checkpoint config.",
    )
    parser.add_argument(
        "--s1-channel-set",
        type=str,
        default=None,
        choices=["all", "vv_vh", "vv", "vh", "vvdiff"],
        help="Override S1 channel set. If omitted, uses checkpoint config.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        choices=["none", "standard", "clip_standard"],
        help="Override model input normalization. If omitted, uses checkpoint config.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Optional limit on dataset rows before sample selection.",
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
        help="Output directory. If omitted, a default QC directory is used.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI. Default: 150",
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


def find_latest_best_checkpoint(output_root: Path) -> Path:
    search_root = output_root / "experiments" / "unet_baseline"

    if not search_root.exists():
        raise FileNotFoundError(
            f"No U-Net experiment directory found: {search_root}"
        )

    candidates = sorted(
        search_root.glob("*/checkpoints/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No best.pt checkpoint found under: {search_root}"
        )

    return candidates[0]


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


def load_python_module(script_name: str, module_name: str):
    current_dir = Path(__file__).resolve().parent
    script_path = current_dir / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(module_name, script_path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def bool_from_any(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}

    return bool(value)


def select_indices(
    df: pd.DataFrame,
    mode: str,
    num_samples: int,
    seed: int,
) -> List[int]:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    all_indices = np.arange(len(df))

    if mode == "first":
        return all_indices[:num_samples].tolist()

    if "is_positive_patch" in df.columns:
        is_positive = df["is_positive_patch"].map(bool_from_any).to_numpy()
    elif "label_positive_percent" in df.columns:
        is_positive = df["label_positive_percent"].astype(float).to_numpy() > 0.0
    else:
        is_positive = np.zeros(len(df), dtype=bool)

    pos_indices = all_indices[is_positive]
    neg_indices = all_indices[~is_positive]

    rng = np.random.default_rng(seed)

    if mode == "positive":
        if len(pos_indices) == 0:
            raise RuntimeError("No positive patches available.")
        n = min(num_samples, len(pos_indices))
        return rng.choice(pos_indices, size=n, replace=False).tolist()

    if mode == "negative":
        if len(neg_indices) == 0:
            raise RuntimeError("No negative patches available.")
        n = min(num_samples, len(neg_indices))
        return rng.choice(neg_indices, size=n, replace=False).tolist()

    if mode == "random":
        n = min(num_samples, len(all_indices))
        return rng.choice(all_indices, size=n, replace=False).tolist()

    if mode == "mixed":
        selected: List[int] = []

        n_pos_target = num_samples // 2
        n_neg_target = num_samples - n_pos_target

        if len(pos_indices) > 0:
            n_pos = min(n_pos_target, len(pos_indices))
            selected.extend(rng.choice(pos_indices, size=n_pos, replace=False).tolist())

        if len(neg_indices) > 0:
            n_neg = min(n_neg_target, len(neg_indices))
            selected.extend(rng.choice(neg_indices, size=n_neg, replace=False).tolist())

        if len(selected) < num_samples:
            selected_set = set(selected)
            remaining = np.array([idx for idx in all_indices if idx not in selected_set])

            if len(remaining) > 0:
                n_extra = min(num_samples - len(selected), len(remaining))
                selected.extend(rng.choice(remaining, size=n_extra, replace=False).tolist())

        return selected

    if mode == "high_positive":
        if "label_positive_percent" not in df.columns:
            raise RuntimeError("Cannot use high_positive mode without label_positive_percent column.")

        ranked = (
            df.reset_index()
            .sort_values("label_positive_percent", ascending=False)
            ["index"]
            .to_numpy()
        )

        return ranked[:num_samples].tolist()

    raise ValueError(f"Unsupported sample mode: {mode}")


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def robust_stretch(
    arr: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    arr = arr.astype("float32", copy=False)

    if valid_mask is not None:
        values = arr[valid_mask]
    else:
        values = arr[np.isfinite(arr)]

    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.zeros_like(arr, dtype="float32")

    low = float(np.percentile(values, p_low))
    high = float(np.percentile(values, p_high))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(values))
        high = float(np.max(values))

    if high <= low:
        return np.zeros_like(arr, dtype="float32")

    out = (arr - low) / (high - low)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0

    return out.astype("float32")


def make_rgb(
    s2: np.ndarray,
    bands: Sequence[int],
    valid_mask: Optional[np.ndarray],
) -> np.ndarray:
    channels = []

    for band_idx in bands:
        channels.append(robust_stretch(s2[band_idx], valid_mask=valid_mask))

    return np.stack(channels, axis=-1)


def make_label_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    overlay = rgb.copy()

    color_img = np.zeros_like(rgb)

    color_img[..., 0] = color[0]
    color_img[..., 1] = color[1]
    color_img[..., 2] = color[2]

    selected = mask > 0.5

    overlay[selected] = (1.0 - alpha) * overlay[selected] + alpha * color_img[selected]

    return np.clip(overlay, 0.0, 1.0)


def make_error_map(
    label: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Return RGB error map:
        true positive  = green
        false positive = red
        false negative = blue
        true negative  = dark gray
        invalid        = black
    """
    h, w = label.shape
    out = np.zeros((h, w, 3), dtype="float32")

    valid_bool = valid > 0.5
    label_bool = label > 0.5
    pred_bool = pred > 0.5

    tn = valid_bool & ~label_bool & ~pred_bool
    tp = valid_bool & label_bool & pred_bool
    fp = valid_bool & ~label_bool & pred_bool
    fn = valid_bool & label_bool & ~pred_bool

    out[tn] = np.array([0.15, 0.15, 0.15], dtype="float32")
    out[tp] = np.array([0.0, 1.0, 0.0], dtype="float32")
    out[fp] = np.array([1.0, 0.0, 0.0], dtype="float32")
    out[fn] = np.array([0.0, 0.25, 1.0], dtype="float32")

    return out


def compute_sample_metrics(
    label: np.ndarray,
    prob: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    valid_bool = valid > 0.5
    label_bool = label > 0.5
    pred_bool = prob >= threshold

    tp = int(np.logical_and.reduce([pred_bool, label_bool, valid_bool]).sum())
    fp = int(np.logical_and.reduce([pred_bool, ~label_bool, valid_bool]).sum())
    fn = int(np.logical_and.reduce([~pred_bool, label_bool, valid_bool]).sum())
    tn = int(np.logical_and.reduce([~pred_bool, ~label_bool, valid_bool]).sum())

    denom_iou = tp + fp + fn
    denom_dice = 2 * tp + fp + fn

    iou = float(tp / denom_iou) if denom_iou > 0 else math.nan
    dice = float((2 * tp) / denom_dice) if denom_dice > 0 else math.nan
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else math.nan
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else math.nan

    valid_pixels = int(valid_bool.sum())
    label_pixels = int(np.logical_and(label_bool, valid_bool).sum())
    pred_pixels = int(np.logical_and(pred_bool, valid_bool).sum())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "valid_pixels": valid_pixels,
        "label_positive_pixels": label_pixels,
        "pred_positive_pixels": pred_pixels,
        "label_positive_percent": 100.0 * label_pixels / max(valid_pixels, 1),
        "pred_positive_percent": 100.0 * pred_pixels / max(valid_pixels, 1),
    }


def add_panel(
    axes: List[Any],
    panel_idx: int,
    image: np.ndarray,
    title: str,
    cmap: Optional[str] = None,
    binary: bool = False,
    contour: Optional[np.ndarray] = None,
    contour_color: str = "yellow",
) -> None:
    ax = axes[panel_idx]

    if binary:
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(image, cmap=cmap)

    if contour is not None and np.any(contour > 0.5):
        ax.contour(
            contour,
            levels=[0.5],
            colors=[contour_color],
            linewidths=0.8,
        )

    ax.set_title(title, fontsize=9)
    ax.axis("off")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_filename(text: str, max_len: int = 170) -> str:
    keep = []

    for char in text:
        if char.isalnum() or char in {"_", "-", "."}:
            keep.append(char)
        else:
            keep.append("_")

    name = "".join(keep)

    if len(name) > max_len:
        name = name[:max_len]

    return name


def batch_input_from_sample(
    sample: Dict[str, Any],
    modality: str,
    device: torch.device,
) -> torch.Tensor:
    tensors = []

    if modality in {"s2", "s2_s1"}:
        tensors.append(sample["s2"].float())

    if modality in {"s1", "s2_s1"}:
        tensors.append(sample["s1"].float())

    x = torch.cat(tensors, dim=0).unsqueeze(0).to(device)

    return x


def visualize_prediction(
    raw_sample: Dict[str, Any],
    model_sample: Dict[str, Any],
    model: torch.nn.Module,
    modality: str,
    device: torch.device,
    threshold: float,
    output_path: Path,
    dpi: int,
    dataset_index: int,
) -> Dict[str, Any]:
    ensure_dir(output_path.parent)

    model.eval()

    with torch.no_grad():
        x = batch_input_from_sample(model_sample, modality, device)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    label = tensor_to_numpy(raw_sample["label"])[0]
    valid = tensor_to_numpy(raw_sample["valid_mask"])[0] > 0.5
    pred = (prob >= threshold).astype("float32")

    metrics = compute_sample_metrics(
        label=label,
        prob=prob,
        valid=valid.astype("float32"),
        threshold=threshold,
    )

    panels: List[Dict[str, Any]] = []

    rgb = None

    if "s2" in raw_sample:
        s2 = tensor_to_numpy(raw_sample["s2"])

        if s2.shape[0] >= 8:
            rgb = make_rgb(s2, bands=[3, 2, 1], valid_mask=valid)
            false_color = make_rgb(s2, bands=[7, 3, 2], valid_mask=valid)

            panels.append(
                {
                    "image": rgb,
                    "title": "S2 RGB B04/B03/B02",
                    "cmap": None,
                    "binary": False,
                    "contour": label,
                    "contour_color": "yellow",
                }
            )

            panels.append(
                {
                    "image": false_color,
                    "title": "S2 False Color B08/B04/B03",
                    "cmap": None,
                    "binary": False,
                    "contour": label,
                    "contour_color": "yellow",
                }
            )

    if "s1" in raw_sample:
        s1 = tensor_to_numpy(raw_sample["s1"])

        if s1.shape[0] >= 1:
            panels.append(
                {
                    "image": robust_stretch(s1[0], valid_mask=valid),
                    "title": "S1 VV_dB",
                    "cmap": "gray",
                    "binary": False,
                    "contour": label,
                    "contour_color": "yellow",
                }
            )

        if s1.shape[0] >= 2:
            panels.append(
                {
                    "image": robust_stretch(s1[1], valid_mask=valid),
                    "title": "S1 VH_dB",
                    "cmap": "gray",
                    "binary": False,
                    "contour": label,
                    "contour_color": "yellow",
                }
            )

        if s1.shape[0] >= 3:
            panels.append(
                {
                    "image": robust_stretch(s1[2], valid_mask=valid),
                    "title": "S1 VV-minus-VH_dB",
                    "cmap": "gray",
                    "binary": False,
                    "contour": label,
                    "contour_color": "yellow",
                }
            )

    panels.append(
        {
            "image": label,
            "title": "Ground truth label",
            "cmap": "gray",
            "binary": True,
            "contour": None,
            "contour_color": "yellow",
        }
    )

    panels.append(
        {
            "image": prob,
            "title": "Predicted probability",
            "cmap": "viridis",
            "binary": True,
            "contour": label,
            "contour_color": "white",
        }
    )

    panels.append(
        {
            "image": pred,
            "title": f"Binary prediction >= {threshold:.2f}",
            "cmap": "gray",
            "binary": True,
            "contour": label,
            "contour_color": "yellow",
        }
    )

    error_map = make_error_map(label=label, pred=pred, valid=valid.astype("float32"))

    panels.append(
        {
            "image": error_map,
            "title": "Error map: TP green, FP red, FN blue",
            "cmap": None,
            "binary": False,
            "contour": None,
            "contour_color": "yellow",
        }
    )

    if rgb is not None:
        pred_overlay = make_label_overlay(
            rgb=rgb,
            mask=pred,
            color=(1.0, 0.0, 0.0),
            alpha=0.45,
        )

        gt_overlay = make_label_overlay(
            rgb=rgb,
            mask=label,
            color=(0.0, 1.0, 0.0),
            alpha=0.45,
        )

        panels.append(
            {
                "image": pred_overlay,
                "title": "Prediction overlay on RGB",
                "cmap": None,
                "binary": False,
                "contour": label,
                "contour_color": "yellow",
            }
        )

        panels.append(
            {
                "image": gt_overlay,
                "title": "Ground truth overlay on RGB",
                "cmap": None,
                "binary": False,
                "contour": pred,
                "contour_color": "red",
            }
        )

    panels.append(
        {
            "image": valid.astype("float32"),
            "title": "Valid mask",
            "cmap": "gray",
            "binary": True,
            "contour": None,
            "contour_color": "yellow",
        }
    )

    n_panels = len(panels)
    ncols = 3
    nrows = int(math.ceil(n_panels / ncols))

    fig, axes_arr = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 4.4 * nrows),
        squeeze=False,
    )

    axes = list(axes_arr.ravel())

    for i, panel in enumerate(panels):
        add_panel(
            axes=axes,
            panel_idx=i,
            image=panel["image"],
            title=panel["title"],
            cmap=panel["cmap"],
            binary=bool(panel["binary"]),
            contour=panel["contour"],
            contour_color=str(panel["contour_color"]),
        )

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    patch_id = str(raw_sample["patch_id"])
    city = str(raw_sample["city"])
    split = str(raw_sample["split"])
    region = str(raw_sample.get("region", ""))

    fig_title = (
        f"idx={dataset_index} | city={city} | split={split} | region={region} | "
        f"IoU={metrics['iou']:.4f} | Dice={metrics['dice']:.4f}\n"
        f"GT%={metrics['label_positive_percent']:.4f} | Pred%={metrics['pred_positive_percent']:.4f} | "
        f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f}\n"
        f"{patch_id}"
    )

    fig.suptitle(fig_title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    output = {
        "dataset_index": dataset_index,
        "patch_id": patch_id,
        "city": city,
        "split": split,
        "region": region,
        "output_path": str(output_path),
        **metrics,
    }

    return output


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))

    checkpoint_path = args.checkpoint or find_latest_best_checkpoint(output_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)

    print("[INFO] Visualize U-Net predictions")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

    # PyTorch >= 2.6 defaults to weights_only=True.
    # Our checkpoint was created locally by 18_train_unet_baseline.py and contains
    # model weights plus training metadata such as pathlib.Path objects.
    # Therefore we explicitly set weights_only=False for trusted local checkpoints.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Backward compatibility for older PyTorch versions that do not support
        # the weights_only argument.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )
    checkpoint_args = checkpoint.get("args", {})

    modality = args.modality or checkpoint_args.get("modality", "s2")
    s2_band_set = args.s2_band_set or checkpoint_args.get("s2_band_set", "all")
    s1_channel_set = args.s1_channel_set or checkpoint_args.get("s1_channel_set", "all")
    normalization = args.normalization or checkpoint_args.get("normalization", "standard")
    threshold = args.threshold if args.threshold is not None else float(checkpoint_args.get("threshold", 0.5))
    base_channels = int(checkpoint_args.get("base_channels", 16))

    if args.filter_set is not None:
        filter_set_path = args.filter_set
    elif checkpoint_args.get("filter_set_path"):
        filter_set_path = Path(str(checkpoint_args["filter_set_path"]))
    else:
        filter_set_path = default_filter_set_path(
            output_root=output_root,
            split_strategy=args.split_strategy,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            filter_set_id=args.filter_set_id,
        )

    if normalization == "none":
        normalization_json_path = None
    elif args.normalization_json is not None:
        normalization_json_path = args.normalization_json
    elif checkpoint_args.get("normalization_json_path"):
        normalization_json_path = Path(str(checkpoint_args["normalization_json_path"]))
    else:
        normalization_json_path = default_normalization_json_path(
            output_root=output_root,
            split_strategy=args.split_strategy,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            filter_set_id=args.filter_set_id,
        )

    if args.output_dir is None:
        experiment_dir = checkpoint_path.parents[1]
        out_dir = (
            experiment_dir
            / "prediction_figures"
            / args.split
            / args.sample_mode
            / f"threshold_{threshold:.2f}"
        )
    else:
        out_dir = args.output_dir

    ensure_dir(out_dir)

    dataset_module = load_python_module(
        "16_build_pytorch_geotiff_dataset.py",
        "favela_pytorch_geotiff_dataset_for_prediction_viz",
    )

    train_module = load_python_module(
        "18_train_unet_baseline.py",
        "favela_unet_baseline_for_prediction_viz",
    )

    DatasetClass = dataset_module.BrazilFavelaGeoTiffDataset
    SmallUNet = train_module.SmallUNet
    infer_input_channels = train_module.infer_input_channels

    input_channels = int(
        checkpoint_args.get(
            "input_channels",
            infer_input_channels(
                modality=modality,
                s2_band_set=s2_band_set,
                s1_channel_set=s1_channel_set,
            ),
        )
    )

    print(f"[INFO] Filter set: {filter_set_path}")
    print(f"[INFO] Normalization JSON: {normalization_json_path}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Sample mode: {args.sample_mode}")
    print(f"[INFO] Num samples: {args.num_samples}")
    print(f"[INFO] Modality: {modality}")
    print(f"[INFO] S2 band set: {s2_band_set}")
    print(f"[INFO] S1 channel set: {s1_channel_set}")
    print(f"[INFO] Normalization: {normalization}")
    print(f"[INFO] Threshold: {threshold}")
    print(f"[INFO] Input channels: {input_channels}")
    print(f"[INFO] Base channels: {base_channels}")
    print(f"[INFO] Output directory: {out_dir}")

    model = SmallUNet(
        in_channels=input_channels,
        base_channels=base_channels,
        out_channels=1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_dataset = DatasetClass(
        patch_csv=filter_set_path,
        split=args.split,
        modality=modality,
        normalization_json=normalization_json_path,
        normalization=normalization,
        s2_band_set=s2_band_set,
        s1_channel_set=s1_channel_set,
        max_patches=args.max_patches,
        treat_all_zero_s2_as_nodata=True,
    )

    raw_dataset = DatasetClass(
        patch_csv=filter_set_path,
        split=args.split,
        modality="s2_s1",
        normalization_json=None,
        normalization="none",
        s2_band_set="all",
        s1_channel_set="all",
        max_patches=args.max_patches,
        treat_all_zero_s2_as_nodata=True,
    )

    if len(model_dataset) != len(raw_dataset):
        raise RuntimeError(
            f"Model dataset and raw dataset have different lengths: "
            f"{len(model_dataset)} vs {len(raw_dataset)}"
        )

    print(f"[INFO] Dataset length: {len(model_dataset)}")

    selected_indices = select_indices(
        df=model_dataset.df,
        mode=args.sample_mode,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    print(f"[INFO] Selected indices: {selected_indices}")

    rows: List[Dict[str, Any]] = []

    for output_i, dataset_idx in enumerate(selected_indices):
        model_sample = model_dataset[dataset_idx]
        raw_sample = raw_dataset[dataset_idx]

        city = str(raw_sample["city"])
        split = str(raw_sample["split"])

        gt_percent = safe_float(raw_sample.get("label_positive_percent", math.nan))

        file_name = safe_filename(
            f"pred_{output_i:03d}__idx_{dataset_idx:05d}__{city}__{split}__gt_{gt_percent:.4f}.png"
        )

        output_path = out_dir / file_name

        row = visualize_prediction(
            raw_sample=raw_sample,
            model_sample=model_sample,
            model=model,
            modality=modality,
            device=device,
            threshold=threshold,
            output_path=output_path,
            dpi=args.dpi,
            dataset_index=dataset_idx,
        )

        rows.append(row)

        print(f"[INFO] Wrote: {output_path}")

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "prediction_visualization_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"[INFO] Wrote summary CSV: {summary_csv}")
    print("[INFO] Prediction visualization completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())