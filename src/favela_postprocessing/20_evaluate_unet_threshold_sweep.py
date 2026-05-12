#!/usr/bin/env python3
"""
Evaluate a trained U-Net checkpoint over multiple probability thresholds.

This script loads a checkpoint produced by:

    18_train_unet_baseline.py

and evaluates IoU, Dice, precision, recall, accuracy, false-positive rate,
false-negative rate, and predicted-positive percentage over many thresholds.

Use validation data to choose the threshold. Do not choose the threshold using
the test set.

Example:

    python src/favela_postprocessing/20_evaluate_unet_threshold_sweep.py `
      --config configs/default.yaml `
      --checkpoint "D:/post_processing_dataset/experiments/unet_baseline/<experiment>/checkpoints/best.pt" `
      --split val `
      --batch-size 1 `
      --num-workers 0 `
      --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm


SCRIPT_NAME = "20_evaluate_unet_threshold_sweep.py"

DEFAULT_SPLIT_STRATEGY = "train_covered_region_test"
DEFAULT_PATCH_SIZE = 512
DEFAULT_STRIDE = 512
DEFAULT_EDGE_MODE = "cover"
DEFAULT_FILTER_SET_ID = "F02_quality_pass"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net predictions across probability thresholds."
    )

    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best.pt or latest.pt. If omitted, latest best.pt is used.",
    )

    parser.add_argument("--filter-set", type=Path, default=None)
    parser.add_argument("--normalization-json", type=Path, default=None)

    parser.add_argument("--filter-set-id", type=str, default=DEFAULT_FILTER_SET_ID)
    parser.add_argument("--split-strategy", type=str, default=DEFAULT_SPLIT_STRATEGY)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--edge-mode", type=str, default=DEFAULT_EDGE_MODE)

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test", "all"],
        help="Split to evaluate. Use val for threshold selection.",
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Explicit thresholds. Default: 0.05, 0.10, ..., 0.95.",
    )

    parser.add_argument("--modality", type=str, default=None, choices=["s2", "s1", "s2_s1"])
    parser.add_argument("--s2-band-set", type=str, default=None, choices=["all", "rgb", "rgb_nir"])
    parser.add_argument("--s1-channel-set", type=str, default=None, choices=["all", "vv_vh", "vv", "vh", "vvdiff"])
    parser.add_argument("--normalization", type=str, default=None, choices=["none", "standard", "clip_standard"])

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=None)

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
        raise FileNotFoundError(f"No U-Net experiment directory found: {search_root}")

    candidates = sorted(
        search_root.glob("*/checkpoints/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No best.pt checkpoint found under: {search_root}")

    return candidates[0]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
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
        raise RuntimeError(f"Could not import script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """
    Load trusted local checkpoint.

    PyTorch >= 2.6 defaults to weights_only=True. Our checkpoints contain
    metadata as well as model weights, so trusted local checkpoints need
    weights_only=False.
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


def make_thresholds(values: Optional[Sequence[float]]) -> List[float]:
    if values is None:
        return [round(float(x), 2) for x in np.arange(0.05, 1.00, 0.05)]

    thresholds = sorted(set(float(v) for v in values))

    for threshold in thresholds:
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError(f"Thresholds must be in (0, 1), got {threshold}")

    return thresholds


def batch_to_input(
    batch: Dict[str, Any],
    modality: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors: List[torch.Tensor] = []

    if modality in {"s2", "s2_s1"}:
        tensors.append(batch["s2"].float())

    if modality in {"s1", "s2_s1"}:
        tensors.append(batch["s1"].float())

    x = torch.cat(tensors, dim=1).to(device, non_blocking=True)
    y = batch["label"].float().to(device, non_blocking=True)
    valid_mask = batch["valid_mask"].float().to(device, non_blocking=True)

    return x, y, valid_mask


class ThresholdAccumulator:
    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.valid_pixels = 0.0
        self.positive_pixels = 0.0
        self.pred_positive_pixels = 0.0

    def update(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        with torch.no_grad():
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
        specificity = self.tn / (self.tn + self.fp + eps)
        false_positive_rate = self.fp / (self.fp + self.tn + eps)
        false_negative_rate = self.fn / (self.fn + self.tp + eps)

        positive_pixel_percent = 100.0 * self.positive_pixels / max(self.valid_pixels, eps)
        pred_positive_pixel_percent = 100.0 * self.pred_positive_pixels / max(self.valid_pixels, eps)

        return {
            "threshold": self.threshold,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "specificity": specificity,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
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


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def markdown_escape(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return "nan"
    except Exception:
        pass

    text = str(value)
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


def df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    columns = list(df.columns)
    header = "| " + " | ".join(markdown_escape(col) for col in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    rows = []
    for _, row in df.iterrows():
        rows.append(
            "| "
            + " | ".join(markdown_escape(row[col]) for col in columns)
            + " |"
        )

    return "\n".join([header, separator] + rows)


def write_markdown_summary(
    path: Path,
    df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)

    best_iou = df.sort_values("iou", ascending=False).iloc[0]
    best_dice = df.sort_values("dice", ascending=False).iloc[0]
    best_balance = df.sort_values("precision_recall_balance", ascending=False).iloc[0]

    lines: List[str] = []

    lines.append("# U-Net Threshold Sweep Summary")
    lines.append("")
    lines.append(f"Generated by `{SCRIPT_NAME}`.")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "This report evaluates a trained U-Net checkpoint across probability thresholds. "
        "Choose thresholds on validation data only, then apply the selected threshold unchanged to test."
    )
    lines.append("")

    lines.append("## Metadata")
    lines.append("")
    metadata_df = pd.DataFrame([{"key": k, "value": v} for k, v in metadata.items()])
    lines.append(df_to_markdown_table(metadata_df))
    lines.append("")

    lines.append("## Best thresholds")
    lines.append("")
    best_df = pd.DataFrame(
        [
            {
                "criterion": "best_iou",
                "threshold": best_iou["threshold"],
                "iou": best_iou["iou"],
                "dice": best_iou["dice"],
                "precision": best_iou["precision"],
                "recall": best_iou["recall"],
                "pred_positive_pixel_percent": best_iou["pred_positive_pixel_percent"],
            },
            {
                "criterion": "best_dice",
                "threshold": best_dice["threshold"],
                "iou": best_dice["iou"],
                "dice": best_dice["dice"],
                "precision": best_dice["precision"],
                "recall": best_dice["recall"],
                "pred_positive_pixel_percent": best_dice["pred_positive_pixel_percent"],
            },
            {
                "criterion": "best_precision_recall_balance",
                "threshold": best_balance["threshold"],
                "iou": best_balance["iou"],
                "dice": best_balance["dice"],
                "precision": best_balance["precision"],
                "recall": best_balance["recall"],
                "pred_positive_pixel_percent": best_balance["pred_positive_pixel_percent"],
            },
        ]
    )

    for col in ["threshold", "iou", "dice", "precision", "recall", "pred_positive_pixel_percent"]:
        best_df[col] = best_df[col].map(lambda x: f"{float(x):.6f}")

    lines.append(df_to_markdown_table(best_df))
    lines.append("")

    lines.append("## Full threshold table")
    lines.append("")

    display_cols = [
        "threshold",
        "iou",
        "dice",
        "precision",
        "recall",
        "accuracy",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "positive_pixel_percent",
        "pred_positive_pixel_percent",
    ]

    display = df[display_cols].copy()

    for col in display_cols:
        display[col] = display[col].map(lambda x: f"{float(x):.6f}")

    lines.append(df_to_markdown_table(display))
    lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- Lower thresholds usually increase recall and false positives.")
    lines.append("- Higher thresholds usually increase precision but can miss true favela pixels.")
    lines.append("- If predicted-positive percentage is much larger than ground-truth-positive percentage, the model overpredicts.")
    lines.append("- If predicted-positive percentage is much smaller than ground-truth-positive percentage, the model underpredicts.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    cfg = load_config(args.config)
    output_root = Path(str(cfg["output_root"]))

    checkpoint_path = args.checkpoint or find_latest_best_checkpoint(output_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)

    print("[INFO] U-Net threshold sweep evaluation")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    checkpoint_args = checkpoint.get("args", {})

    modality = args.modality or checkpoint_args.get("modality", "s2")
    s2_band_set = args.s2_band_set or checkpoint_args.get("s2_band_set", "all")
    s1_channel_set = args.s1_channel_set or checkpoint_args.get("s1_channel_set", "all")
    normalization = args.normalization or checkpoint_args.get("normalization", "standard")
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

    thresholds = make_thresholds(args.thresholds)

    if args.output_dir is None:
        experiment_dir = checkpoint_path.parents[1]
        out_dir = experiment_dir / "threshold_sweep" / args.split
    else:
        out_dir = args.output_dir

    ensure_dir(out_dir)

    dataset_module = load_python_module(
        "16_build_pytorch_geotiff_dataset.py",
        "favela_pytorch_geotiff_dataset_for_threshold_sweep",
    )

    train_module = load_python_module(
        "18_train_unet_baseline.py",
        "favela_unet_baseline_for_threshold_sweep",
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
    print(f"[INFO] Thresholds: {thresholds}")
    print(f"[INFO] Modality: {modality}")
    print(f"[INFO] S2 band set: {s2_band_set}")
    print(f"[INFO] S1 channel set: {s1_channel_set}")
    print(f"[INFO] Normalization: {normalization}")
    print(f"[INFO] Input channels: {input_channels}")
    print(f"[INFO] Base channels: {base_channels}")
    print(f"[INFO] Output directory: {out_dir}")

    dataset = DatasetClass(
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

    print(f"[INFO] Dataset length: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SmallUNet(
        in_channels=input_channels,
        base_channels=base_channels,
        out_channels=1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accumulators = [ThresholdAccumulator(t) for t in thresholds]

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            x, y, valid_mask = batch_to_input(batch, modality, device)
            logits = model(x)
            probs = torch.sigmoid(logits)

            for acc in accumulators:
                acc.update(
                    probs=probs,
                    targets=y,
                    valid_mask=valid_mask,
                )

    rows = [acc.compute() for acc in accumulators]
    df = pd.DataFrame(rows)

    df["precision_recall_balance"] = 1.0 - np.abs(df["precision"] - df["recall"])
    df = df.sort_values("threshold").reset_index(drop=True)

    best_iou_row = df.sort_values("iou", ascending=False).iloc[0].to_dict()
    best_dice_row = df.sort_values("dice", ascending=False).iloc[0].to_dict()

    metrics_csv = out_dir / "threshold_sweep_metrics.csv"
    metrics_json = out_dir / "threshold_sweep_metrics.json"
    summary_md = out_dir / "threshold_sweep_summary.md"

    df.to_csv(metrics_csv, index=False)

    metadata = {
        "script": SCRIPT_NAME,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "dataset_length": len(dataset),
        "filter_set_path": str(filter_set_path),
        "normalization_json_path": str(normalization_json_path) if normalization_json_path else None,
        "modality": modality,
        "s2_band_set": s2_band_set,
        "s1_channel_set": s1_channel_set,
        "normalization": normalization,
        "input_channels": input_channels,
        "base_channels": base_channels,
        "batch_size": args.batch_size,
        "max_patches": args.max_patches,
        "device": str(device),
        "important_note": "Select threshold on validation only. Apply unchanged to test.",
    }

    json_data = {
        "metadata": metadata,
        "best_iou": best_iou_row,
        "best_dice": best_dice_row,
        "thresholds": rows,
    }

    write_json(metrics_json, json_data)
    write_markdown_summary(summary_md, df, metadata)

    print(f"[INFO] Wrote CSV: {metrics_csv}")
    print(f"[INFO] Wrote JSON: {metrics_json}")
    print(f"[INFO] Wrote Markdown: {summary_md}")

    print("[INFO] Best by IoU:")
    print(
        pd.DataFrame([best_iou_row])[
            [
                "threshold",
                "iou",
                "dice",
                "precision",
                "recall",
                "positive_pixel_percent",
                "pred_positive_pixel_percent",
            ]
        ].to_string(index=False)
    )

    print("[INFO] Best by Dice:")
    print(
        pd.DataFrame([best_dice_row])[
            [
                "threshold",
                "iou",
                "dice",
                "precision",
                "recall",
                "positive_pixel_percent",
                "pred_positive_pixel_percent",
            ]
        ].to_string(index=False)
    )

    print("[INFO] Threshold sweep evaluation completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())