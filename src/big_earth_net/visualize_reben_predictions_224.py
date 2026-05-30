#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
visualize_reben_predictions_224.py

Main objective
--------------
Generate qualitative prediction previews for a trained reBEN ResNet18 + UPerNet
S1/S2 favela segmentation model.

The script loads a trained checkpoint, evaluates patch-level metrics, then saves
visual panels showing:

    - Sentinel-2 RGB
    - Sentinel-2 false colour
    - Sentinel-1 VV
    - Sentinel-1 VH
    - ground-truth mask
    - prediction probability map
    - binary prediction
    - error map: TP / FP / FN

This is a diagnostic script. It does not train a model.

Recommended command
-------------------
python src\\big_earth_net\\visualize_reben_predictions_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --run-dir "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired/experiments/big_earth_net/reben_resnet18_upernet_s1s2_train_region_covered_ps224/reben_resnet18_upernet_s1s2_train_region_covered_epochs30_bs2_acc4_fullfinetune_posw10" `
  --checkpoint best `
  --split test `
  --batch-size 8 `
  --num-workers 0 `
  --n-per-category 12 `
  --overwrite

Outputs
-------
<run-dir>/visual_predictions/<split>_<checkpoint>_thrXXXX/

    per_patch_metrics_<split>.csv
    selected_patches_<split>.csv
    diagnostic_report_<split>.md
    panels/
        random/
        best_iou/
        worst_iou/
        high_false_positive/
        high_false_negative/
        overprediction/
        underprediction/
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise SystemExit(
        "[ERROR] PyTorch is required.\n"
        f"Original error: {exc}"
    )

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from train_reben_resnet18_upernet_s1s2_224 import (
        DEFAULT_S1_BAND_INDICES,
        DEFAULT_S2_BAND_INDICES_REBEN,
        RebenResNet18UPerNet,
        RebenS1S2SegmentationDataset,
        choose_device,
        clear_torch_memory,
        count_parameters,
        jsonable,
        parse_int_list,
        path_to_str,
    )
except ImportError as exc:
    raise SystemExit(
        "[ERROR] Could not import objects from train_reben_resnet18_upernet_s1s2_224.py.\n"
        "Make sure this visualization script is in:\n"
        "    src/big_earth_net/\n\n"
        f"Original error: {exc}"
    )


# ---------------------------------------------------------------------
# Logging
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        fail(
            "Output directory already exists and is not empty:\n"
            f"{path_to_str(path)}\n\n"
            "Use --overwrite to replace/update the visual outputs."
        )
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Paths and checkpoint handling
# ---------------------------------------------------------------------

def default_split_dir(instance_root: Path) -> Path:
    return (
        instance_root
        / "metadata"
        / "big_earth_net"
        / "region_balanced_city_split_ps224_st112_cover"
    )


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    text = str(checkpoint).strip()

    if text.lower() == "best":
        return run_dir / "checkpoints" / "best.pt"

    if text.lower() == "latest":
        return run_dir / "checkpoints" / "latest.pt"

    path = Path(text)
    if path.exists():
        return path

    candidate = run_dir / "checkpoints" / text
    if candidate.exists():
        return candidate

    fail(
        "Could not resolve checkpoint:\n"
        f"  checkpoint argument: {checkpoint}\n"
        f"  tried as path:       {path_to_str(path)}\n"
        f"  tried in run dir:    {path_to_str(candidate)}"
    )


def checkpoint_label(checkpoint: str, checkpoint_path: Path) -> str:
    text = str(checkpoint).strip().lower()
    if text in {"best", "latest"}:
        return text
    return checkpoint_path.stem


def load_training_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.json"

    if not config_path.exists():
        warn(f"config.json not found:\n{path_to_str(config_path)}")
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested_arg(config: Dict[str, Any], ckpt: Dict[str, Any], name: str, default: Any) -> Any:
    if "args" in ckpt and isinstance(ckpt["args"], dict) and name in ckpt["args"]:
        return ckpt["args"][name]

    if "args" in config and isinstance(config["args"], dict) and name in config["args"]:
        return config["args"][name]

    return default


def load_checkpoint_file(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        fail(f"Checkpoint file does not exist:\n{path_to_str(path)}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_from_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
) -> Tuple[RebenResNet18UPerNet, Dict[str, Any], Dict[str, Any]]:
    config = load_training_config(run_dir)
    ckpt = load_checkpoint_file(checkpoint_path, device=device)

    if "model_state_dict" not in ckpt:
        fail(f"Checkpoint does not contain model_state_dict:\n{path_to_str(checkpoint_path)}")

    decoder_channels = int(get_nested_arg(config, ckpt, "decoder_channels", 128))
    ppm_channels = int(get_nested_arg(config, ckpt, "ppm_channels", 32))
    dropout = float(get_nested_arg(config, ckpt, "dropout", 0.10))
    normalization = str(get_nested_arg(config, ckpt, "normalization", "per_sample_channel"))
    out_indices = get_nested_arg(config, ckpt, "out_indices", [1, 2, 3, 4])

    if isinstance(out_indices, str):
        out_indices = [int(x) for x in out_indices.replace(",", " ").split() if x]

    model = RebenResNet18UPerNet(
        decoder_channels=decoder_channels,
        ppm_channels=ppm_channels,
        dropout=dropout,
        normalization=normalization,
        out_indices=tuple(int(x) for x in out_indices),
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    info = {
        "checkpoint_path": path_to_str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch", None),
        "decoder_channels": decoder_channels,
        "ppm_channels": ppm_channels,
        "dropout": dropout,
        "normalization": normalization,
        "out_indices": out_indices,
        "feature_channels": model.feature_channels,
        "model_counts": count_parameters(model),
    }

    return model, ckpt, info


# ---------------------------------------------------------------------
# Threshold handling
# ---------------------------------------------------------------------

def extract_threshold_from_final_summary(run_dir: Path) -> Optional[float]:
    final_summary_path = run_dir / "final_summary.json"

    if not final_summary_path.exists():
        return None

    try:
        with final_summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return None

    selected = summary.get("selected_threshold")

    if isinstance(selected, dict) and "threshold" in selected:
        return float(selected["threshold"])

    if isinstance(selected, (int, float)):
        return float(selected)

    return None


def resolve_threshold(args: argparse.Namespace, run_dir: Path) -> float:
    if args.threshold is not None:
        return float(args.threshold)

    from_summary = extract_threshold_from_final_summary(run_dir)
    if from_summary is not None:
        log("INFO", f"Using threshold from final_summary.json: {from_summary}")
        return float(from_summary)

    warn("Could not find selected threshold in final_summary.json. Falling back to threshold 0.5.")
    return 0.5


# ---------------------------------------------------------------------
# Dataset and dataloader
# ---------------------------------------------------------------------

def split_csv_path(split_dir: Path, split: str) -> Path:
    return split_dir / f"{split}.csv"


def make_loader(
    dataset: RebenS1S2SegmentationDataset,
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
# Metrics
# ---------------------------------------------------------------------

def compute_patch_metrics_from_counts(
    tp: float,
    fp: float,
    fn: float,
    tn: float,
    n_pixels: float,
) -> Dict[str, float]:
    eps = 1e-8

    iou_favela = tp / (tp + fp + fn + eps)
    dice_favela = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    precision_favela = tp / (tp + fp + eps)
    recall_favela = tp / (tp + fn + eps)

    tp_no = tn
    fp_no = fn
    fn_no = fp

    iou_no_favela = tp_no / (tp_no + fp_no + fn_no + eps)
    dice_no_favela = (2.0 * tp_no) / (2.0 * tp_no + fp_no + fn_no + eps)
    precision_no_favela = tp_no / (tp_no + fp_no + eps)
    recall_no_favela = tp_no / (tp_no + fn_no + eps)

    macro_iou = 0.5 * (iou_favela + iou_no_favela)
    macro_dice = 0.5 * (dice_favela + dice_no_favela)

    pred_pos = tp + fp
    gt_pos = tp + fn

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),

        "iou_favela": float(iou_favela),
        "dice_favela": float(dice_favela),
        "precision_favela": float(precision_favela),
        "recall_favela": float(recall_favela),

        "iou_no_favela": float(iou_no_favela),
        "dice_no_favela": float(dice_no_favela),
        "precision_no_favela": float(precision_no_favela),
        "recall_no_favela": float(recall_no_favela),

        "macro_iou": float(macro_iou),
        "macro_dice": float(macro_dice),

        "pred_pos_pct": float(100.0 * pred_pos / max(eps, n_pixels)),
        "gt_pos_pct": float(100.0 * gt_pos / max(eps, n_pixels)),
        "fp_pct": float(100.0 * fp / max(eps, n_pixels)),
        "fn_pct": float(100.0 * fn / max(eps, n_pixels)),
        "pred_minus_gt_pct": float(100.0 * (pred_pos - gt_pos) / max(eps, n_pixels)),
        "gt_minus_pred_pct": float(100.0 * (gt_pos - pred_pos) / max(eps, n_pixels)),
    }


@torch.no_grad()
def evaluate_per_patch(
    model: RebenResNet18UPerNet,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    max_patches: Optional[int],
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, Any]] = []

    seen = 0

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="evaluating patches", unit="batch")

    for batch in iterator:
        x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
        masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)

        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs >= float(threshold)).float()
        targets = (masks >= 0.5).float()

        b = int(x.shape[0])

        for i in range(b):
            if max_patches is not None and seen >= int(max_patches):
                break

            pred_i = preds[i]
            target_i = targets[i]

            tp = float((pred_i * target_i).sum().item())
            fp = float((pred_i * (1.0 - target_i)).sum().item())
            fn = float(((1.0 - pred_i) * target_i).sum().item())
            tn = float(((1.0 - pred_i) * (1.0 - target_i)).sum().item())
            n_pixels = float(target_i.numel())

            metric_row = compute_patch_metrics_from_counts(tp, fp, fn, tn, n_pixels)

            rows.append(
                {
                    "dataset_index": seen,
                    "patch_id": batch["patch_id"][i],
                    "city": batch["city"][i],
                    "region": batch["region"][i],
                    **metric_row,
                }
            )

            seen += 1

        del x, masks, logits, probs, preds, targets

        if max_patches is not None and seen >= int(max_patches):
            break

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Raster visualization helpers
# ---------------------------------------------------------------------

def read_window(
    raster_path: Path,
    row_off: int,
    col_off: int,
    patch_size: int,
    band_indices: Sequence[int],
    fill_value: float = 0.0,
) -> np.ndarray:
    with rasterio.open(raster_path) as src:
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


def robust_stretch(arr: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    x = arr.astype(np.float32)
    finite = np.isfinite(x)

    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)

    low = np.percentile(x[finite], low_pct)
    high = np.percentile(x[finite], high_pct)

    if high <= low:
        return np.zeros_like(x, dtype=np.float32)

    x = (x - low) / (high - low)
    return np.clip(x, 0.0, 1.0)


def make_rgb(chw: np.ndarray) -> np.ndarray:
    channels = [robust_stretch(chw[i]) for i in range(chw.shape[0])]
    return np.stack(channels, axis=-1)


def resolve_existing_path(path_value: Any, instance_root: Path) -> Path:
    raw = str(path_value).strip().replace("\\", "/")
    p = Path(raw)

    if p.exists():
        return p

    if not p.is_absolute():
        candidate = instance_root / raw
        if candidate.exists():
            return candidate

    fail(
        "Raster path does not exist:\n"
        f"  original: {raw}\n"
        f"  tried:    {path_to_str(p)}"
    )


def make_error_rgb(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = (gt > 0.5)
    pred = (pred > 0.5)

    h, w = gt.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    tn = (~gt) & (~pred)
    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)

    out[tn] = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    out[tp] = np.array([0.0, 0.75, 0.0], dtype=np.float32)
    out[fp] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out[fn] = np.array([0.0, 0.25, 1.0], dtype=np.float32)

    return out


def get_row_for_patch(dataset: RebenS1S2SegmentationDataset, patch_id: str) -> pd.Series:
    matches = dataset.df[dataset.df["patch_id"].astype(str) == str(patch_id)]

    if matches.empty:
        fail(f"Patch id not found in dataset dataframe: {patch_id}")

    return matches.iloc[0]


@torch.no_grad()
def predict_single_patch(
    model: RebenResNet18UPerNet,
    dataset: RebenS1S2SegmentationDataset,
    patch_id: str,
    device: torch.device,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matches = dataset.df.index[dataset.df["patch_id"].astype(str) == str(patch_id)].tolist()

    if not matches:
        fail(f"Patch id not found in dataset: {patch_id}")

    idx = int(matches[0])
    sample = dataset[idx]

    x = sample["x"].unsqueeze(0).to(device=device, dtype=torch.float32)
    gt = sample["mask"].squeeze(0).numpy().astype(np.float32)

    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    pred = (prob >= float(threshold)).astype(np.float32)

    return gt, prob, pred


def save_patch_panel(
    model: RebenResNet18UPerNet,
    dataset: RebenS1S2SegmentationDataset,
    patch_metrics_row: pd.Series,
    instance_root: Path,
    output_path: Path,
    device: torch.device,
    threshold: float,
    patch_size: int,
) -> None:
    patch_id = str(patch_metrics_row["patch_id"])
    row = get_row_for_patch(dataset, patch_id)

    row_start = int(row["row_start"])
    col_start = int(row["col_start"])

    optical_path = resolve_existing_path(row["optical_path"], instance_root)
    sar_path = resolve_existing_path(row["sar_path"], instance_root)

    # Original S2 stack order:
    # B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
    s2_rgb_chw = read_window(
        optical_path,
        row_off=row_start,
        col_off=col_start,
        patch_size=patch_size,
        band_indices=[4, 3, 2],
        fill_value=0.0,
    )

    s2_false_chw = read_window(
        optical_path,
        row_off=row_start,
        col_off=col_start,
        patch_size=patch_size,
        band_indices=[8, 4, 3],
        fill_value=0.0,
    )

    s1_chw = read_window(
        sar_path,
        row_off=row_start,
        col_off=col_start,
        patch_size=patch_size,
        band_indices=[1, 2],
        fill_value=0.0,
    )

    gt, prob, pred = predict_single_patch(
        model=model,
        dataset=dataset,
        patch_id=patch_id,
        device=device,
        threshold=threshold,
    )

    s2_rgb = make_rgb(s2_rgb_chw)
    s2_false = make_rgb(s2_false_chw)
    vv = robust_stretch(s1_chw[0])
    vh = robust_stretch(s1_chw[1])
    error_rgb = make_error_rgb(gt, pred)

    ensure_dir(output_path.parent)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(s2_rgb)
    axes[0, 0].set_title("S2 RGB\nB04/B03/B02")

    axes[0, 1].imshow(s2_false)
    axes[0, 1].set_title("S2 false colour\nB08/B04/B03")

    axes[0, 2].imshow(vv, cmap="gray")
    axes[0, 2].set_title("S1 VV")

    axes[0, 3].imshow(vh, cmap="gray")
    axes[0, 3].set_title("S1 VH")

    axes[1, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Ground truth")

    im = axes[1, 1].imshow(prob, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title("Predicted probability")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title(f"Binary prediction\nthreshold={threshold:.3f}")

    axes[1, 3].imshow(error_rgb)
    axes[1, 3].set_title("Error map\nGreen=TP, Red=FP, Blue=FN")

    for ax in axes.ravel():
        ax.axis("off")

    title = (
        f"patch={patch_id} | city={patch_metrics_row['city']} | region={patch_metrics_row['region']}\n"
        f"IoU_favela={patch_metrics_row['iou_favela']:.4f}, "
        f"Dice={patch_metrics_row['dice_favela']:.4f}, "
        f"P={patch_metrics_row['precision_favela']:.4f}, "
        f"R={patch_metrics_row['recall_favela']:.4f}, "
        f"pred+={patch_metrics_row['pred_pos_pct']:.2f}%, "
        f"gt+={patch_metrics_row['gt_pos_pct']:.2f}%"
    )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------
# Patch selection
# ---------------------------------------------------------------------

def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def select_diagnostic_patches(
    metrics_df: pd.DataFrame,
    n_per_category: int,
    seed: int,
    min_gt_pos_pct: float,
) -> pd.DataFrame:
    selected_parts: List[pd.DataFrame] = []

    df = metrics_df.copy()
    positive_df = df[df["gt_pos_pct"] >= float(min_gt_pos_pct)].copy()

    categories = []

    random_df = sample_rows(df, n_per_category, seed)
    random_df["category"] = "random"
    categories.append(random_df)

    if not positive_df.empty:
        best = positive_df.sort_values("iou_favela", ascending=False).head(n_per_category).copy()
        best["category"] = "best_iou"
        categories.append(best)

        worst = positive_df.sort_values("iou_favela", ascending=True).head(n_per_category).copy()
        worst["category"] = "worst_iou"
        categories.append(worst)

        high_fn = positive_df.sort_values("fn_pct", ascending=False).head(n_per_category).copy()
        high_fn["category"] = "high_false_negative"
        categories.append(high_fn)

        under = positive_df.sort_values("gt_minus_pred_pct", ascending=False).head(n_per_category).copy()
        under["category"] = "underprediction"
        categories.append(under)

    high_fp = df.sort_values("fp_pct", ascending=False).head(n_per_category).copy()
    high_fp["category"] = "high_false_positive"
    categories.append(high_fp)

    over = df.sort_values("pred_minus_gt_pct", ascending=False).head(n_per_category).copy()
    over["category"] = "overprediction"
    categories.append(over)

    for part in categories:
        selected_parts.append(part)

    if not selected_parts:
        return pd.DataFrame()

    selected = pd.concat(selected_parts, axis=0, ignore_index=True)

    role_cols = ["category", "patch_id", "city", "region"]
    other_cols = [c for c in selected.columns if c not in role_cols]

    return selected[role_cols + other_cols].reset_index(drop=True)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def markdown_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows).copy()

    if df.empty:
        return "_No rows._"

    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_report(
    path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint_path: Path,
    model_info: Dict[str, Any],
    threshold: float,
    metrics_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> None:
    lines: List[str] = []

    lines.append(f"# reBEN Prediction Visualization Report: {args.split}")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Generate qualitative prediction panels and per-patch metrics in order to diagnose "
        "why the reBEN ResNet18 + UPerNet S1/S2 model achieves limited favela IoU."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Instance root: `{args.instance_root}`")
    lines.append(f"- Run directory: `{args.run_dir}`")
    lines.append(f"- Checkpoint: `{path_to_str(checkpoint_path)}`")
    lines.append(f"- Split: `{args.split}`")
    lines.append(f"- Threshold: `{threshold}`")
    lines.append(f"- Output directory: `{path_to_str(output_dir)}`")
    lines.append("")
    lines.append("## Model Info")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(jsonable(model_info), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Per-Patch Metric Summary")
    lines.append("")

    summary_cols = [
        "iou_favela",
        "dice_favela",
        "precision_favela",
        "recall_favela",
        "iou_no_favela",
        "macro_iou",
        "pred_pos_pct",
        "gt_pos_pct",
        "fp_pct",
        "fn_pct",
    ]

    summary = metrics_df[summary_cols].describe().reset_index()
    lines.append(markdown_table(summary))
    lines.append("")
    lines.append("## Mean Metrics by City")
    lines.append("")

    city_summary = (
        metrics_df.groupby(["region", "city"])
        .agg(
            n_patches=("patch_id", "count"),
            mean_iou_favela=("iou_favela", "mean"),
            mean_dice_favela=("dice_favela", "mean"),
            mean_precision_favela=("precision_favela", "mean"),
            mean_recall_favela=("recall_favela", "mean"),
            mean_pred_pos_pct=("pred_pos_pct", "mean"),
            mean_gt_pos_pct=("gt_pos_pct", "mean"),
            mean_fp_pct=("fp_pct", "mean"),
            mean_fn_pct=("fn_pct", "mean"),
        )
        .reset_index()
        .sort_values("mean_iou_favela", ascending=True)
    )

    lines.append(markdown_table(city_summary))
    lines.append("")
    lines.append("## Selected Diagnostic Patches")
    lines.append("")
    selected_cols = [
        "category",
        "patch_id",
        "city",
        "region",
        "iou_favela",
        "dice_favela",
        "precision_favela",
        "recall_favela",
        "pred_pos_pct",
        "gt_pos_pct",
        "fp_pct",
        "fn_pct",
    ]
    lines.append(markdown_table(selected_df[selected_cols], max_rows=200))
    lines.append("")
    lines.append("## How to Interpret the Panels")
    lines.append("")
    lines.append("- Green pixels in the error map are true-positive favela detections.")
    lines.append("- Red pixels are false positives: predicted favela, but ground truth is no-favela.")
    lines.append("- Blue pixels are false negatives: true favela pixels missed by the model.")
    lines.append("- If red dominates, the model is overpredicting favelas.")
    lines.append("- If blue dominates, the model is missing real favela areas.")
    lines.append("- If prediction is spatially shifted, check alignment and window reading.")
    lines.append("- If prediction follows dense urban texture outside labels, hard-negative mining may be needed.")
    lines.append("- If prediction roughly matches but boundaries are poor, boundary noise or 10 m resolution may be limiting.")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    instance_root = Path(args.instance_root)
    run_dir = Path(args.run_dir)
    split_dir = Path(args.split_dir) if args.split_dir else default_split_dir(instance_root)

    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint)
    ckpt_lbl = checkpoint_label(args.checkpoint, checkpoint_path)

    device = choose_device(force_cpu=bool(args.force_cpu), device_index=int(args.device_index))
    threshold = resolve_threshold(args, run_dir)

    threshold_label = f"thr{threshold:.3f}".replace(".", "p")
    output_dir = run_dir / "visual_predictions" / f"{args.split}_{ckpt_lbl}_{threshold_label}"

    ensure_output_dir(output_dir, overwrite=bool(args.overwrite))

    banner("Visualize reBEN predictions")

    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Run dir:       {path_to_str(run_dir)}")
    log("INFO", f"Split dir:     {path_to_str(split_dir)}")
    log("INFO", f"Split:         {args.split}")
    log("INFO", f"Checkpoint:    {path_to_str(checkpoint_path)}")
    log("INFO", f"Threshold:     {threshold}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")
    log("INFO", f"Device:        {device}")

    model, ckpt, model_info = build_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        device=device,
    )

    s1_band_indices = parse_int_list(args.s1_band_indices, DEFAULT_S1_BAND_INDICES)
    s2_band_indices = parse_int_list(args.s2_band_indices, DEFAULT_S2_BAND_INDICES_REBEN)

    split_csv = split_csv_path(split_dir, args.split)

    dataset = RebenS1S2SegmentationDataset(
        csv_path=split_csv,
        instance_root=instance_root,
        patch_size=int(args.patch_size),
        s1_band_indices=s1_band_indices,
        s2_band_indices=s2_band_indices,
    )

    loader = make_loader(
        dataset=dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=(not args.no_pin_memory and device.type == "cuda"),
    )

    log("INFO", f"Dataset patches: {len(dataset):,}")

    metrics_df = evaluate_per_patch(
        model=model,
        loader=loader,
        device=device,
        threshold=threshold,
        max_patches=args.max_patches,
    )

    if metrics_df.empty:
        fail("No patch metrics were computed.")

    metrics_path = output_dir / f"per_patch_metrics_{args.split}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log("OK", f"Per-patch metrics saved:\n{path_to_str(metrics_path)}")

    selected_df = select_diagnostic_patches(
        metrics_df=metrics_df,
        n_per_category=int(args.n_per_category),
        seed=int(args.seed),
        min_gt_pos_pct=float(args.min_gt_pos_pct),
    )

    selected_path = output_dir / f"selected_patches_{args.split}.csv"
    selected_df.to_csv(selected_path, index=False)
    log("OK", f"Selected patches saved:\n{path_to_str(selected_path)}")

    panels_dir = output_dir / "panels"
    ensure_dir(panels_dir)

    log("STEP", "Saving diagnostic panels")

    if tqdm is not None:
        iterator = tqdm(selected_df.iterrows(), total=len(selected_df), desc="saving panels", unit="panel")
    else:
        iterator = selected_df.iterrows()

    saved_count = 0

    for idx, row in iterator:
        category = str(row["category"])
        patch_id = str(row["patch_id"])
        city = str(row["city"])
        safe_city = city.replace("/", "_").replace("\\", "_").replace(" ", "_")

        filename = (
            f"{saved_count:04d}_"
            f"{category}_"
            f"{safe_city}_"
            f"{patch_id}_"
            f"iou{float(row['iou_favela']):.3f}_"
            f"pred{float(row['pred_pos_pct']):.2f}_"
            f"gt{float(row['gt_pos_pct']):.2f}.png"
        )

        output_path = panels_dir / category / filename

        try:
            save_patch_panel(
                model=model,
                dataset=dataset,
                patch_metrics_row=row,
                instance_root=instance_root,
                output_path=output_path,
                device=device,
                threshold=threshold,
                patch_size=int(args.patch_size),
            )
            saved_count += 1
        except Exception as exc:
            warn(f"Failed to save panel for patch {patch_id}: {exc}")
            traceback.print_exc()

    log("OK", f"Saved {saved_count} diagnostic panels.")

    report_path = output_dir / f"diagnostic_report_{args.split}.md"

    write_report(
        path=report_path,
        args=args,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        model_info=model_info,
        threshold=threshold,
        metrics_df=metrics_df,
        selected_df=selected_df,
    )

    write_json(
        output_dir / "visualization_config.json",
        {
            "instance_root": path_to_str(instance_root),
            "run_dir": path_to_str(run_dir),
            "split_dir": path_to_str(split_dir),
            "split": args.split,
            "checkpoint_path": path_to_str(checkpoint_path),
            "threshold": threshold,
            "output_dir": path_to_str(output_dir),
            "model_info": model_info,
            "n_metrics_rows": int(len(metrics_df)),
            "n_selected_rows": int(len(selected_df)),
            "n_saved_panels": int(saved_count),
        },
    )

    banner("Completed")
    log("OK", f"Output directory:\n{path_to_str(output_dir)}")
    log("OK", f"Report:\n{path_to_str(report_path)}")

    clear_torch_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize predictions from a trained reBEN ResNet18 + UPerNet S1/S2 favela segmentation model."
    )

    parser.add_argument(
        "--instance-root",
        required=True,
        help="Dataset instance root.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training run directory containing config.json and checkpoints/.",
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Optional split directory. Default: instance-root/metadata/big_earth_net/region_balanced_city_split_ps224_st112_cover",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint to load: best, latest, explicit .pt path, or filename inside checkpoints/.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Split to visualize.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binary threshold. If omitted, tries final_summary.json selected threshold, then falls back to 0.5.",
    )

    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--s1-band-indices", default=None)
    parser.add_argument("--s2-band-indices", default=None)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-patches", type=int, default=None)

    parser.add_argument("--n-per-category", type=int, default=12)
    parser.add_argument(
        "--min-gt-pos-pct",
        type=float,
        default=0.05,
        help="Minimum GT favela percentage for best/worst/high-FN positive-patch categories.",
    )

    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())