#!/usr/bin/env python3
"""
Visualize PyTorch GeoTIFF dataset samples for QC.

Purpose
-------
This script uses the PyTorch GeoTIFF Dataset from:

    16_build_pytorch_geotiff_dataset.py

and generates PNG visualizations for sampled patches.

It does NOT train a model.
It does NOT export H5 files.
It does NOT duplicate the dataset.

For each selected patch, it visualizes:
    - Sentinel-2 RGB preview: B04/B03/B02
    - Sentinel-2 false-color preview: B08/B04/B03
    - Sentinel-1 VV_dB
    - Sentinel-1 VH_dB
    - Sentinel-1 VV_minus_VH_dB
    - binary label mask
    - label overlay on S2 RGB
    - valid mask

Important visualization fix
---------------------------
Binary panels such as label mask and valid mask are displayed with fixed
vmin=0 and vmax=1. Without this, matplotlib can render constant masks
misleadingly.

Examples
--------
Visualize mixed test patches:

    python3 src/favela_postprocessing/17_visualize_pytorch_dataset_samples.py \
        --config configs/default.yaml \
        --split test \
        --sample-mode mixed \
        --num-samples 8

Visualize positive validation patches:

    python3 src/favela_postprocessing/17_visualize_pytorch_dataset_samples.py \
        --config configs/default.yaml \
        --split val \
        --sample-mode positive \
        --num-samples 8
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SCRIPT_NAME = "17_visualize_pytorch_dataset_samples.py"

DEFAULT_SPLIT_STRATEGY = "train_covered_region_test"
DEFAULT_PATCH_SIZE = 512
DEFAULT_STRIDE = 512
DEFAULT_EDGE_MODE = "cover"
DEFAULT_FILTER_SET_ID = "F02_quality_pass"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize samples from the PyTorch GeoTIFF dataset."
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
        help="Explicit normalization JSON path. Only needed if normalization != none.",
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
        default="train",
        choices=["train", "val", "test", "all"],
        help="Dataset split to visualize. Default: train",
    )
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="mixed",
        choices=["first", "random", "positive", "negative", "mixed"],
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
        help="Random seed for sample selection. Default: 42",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="none",
        choices=["none", "standard", "clip_standard"],
        help="Normalization mode used when reading samples. Default: none for visual QC.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. If omitted, a default QC figure directory is used.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI. Default: 150",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Optional maximum number of patches loaded into the dataset.",
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


def default_output_dir(
    output_root: Path,
    split_strategy: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
    filter_set_id: str,
    split: str,
    sample_mode: str,
) -> Path:
    suffix = default_suffix(split_strategy, patch_size, stride, edge_mode)

    return (
        output_root
        / "qc"
        / "figures"
        / "pytorch_dataset_samples"
        / suffix
        / filter_set_id
        / split
        / sample_mode
    )


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

    rng = np.random.default_rng(seed)

    if "is_positive_patch" in df.columns:
        is_positive = df["is_positive_patch"].map(bool_from_any).to_numpy()
    elif "label_positive_percent" in df.columns:
        is_positive = df["label_positive_percent"].astype(float).to_numpy() > 0.0
    else:
        is_positive = np.zeros(len(df), dtype=bool)

    pos_indices = all_indices[is_positive]
    neg_indices = all_indices[~is_positive]

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
    """
    Stretch array to [0, 1] using robust percentiles.
    """
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
    """
    Make RGB display from S2 array [C, H, W].

    Bands are zero-based indices:
        B04/B03/B02 = [3, 2, 1]
        B08/B04/B03 = [7, 3, 2]
    """
    channels = []

    for band_idx in bands:
        channels.append(robust_stretch(s2[band_idx], valid_mask=valid_mask))

    return np.stack(channels, axis=-1)


def make_filled_label_overlay(
    rgb: np.ndarray,
    label: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay binary label mask on RGB using red fill.
    """
    overlay = rgb.copy()
    mask = label > 0.5

    red = np.zeros_like(rgb)
    red[..., 0] = 1.0

    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * red[mask]

    return np.clip(overlay, 0.0, 1.0)


def add_panel(
    axes: List[Any],
    panel_idx: int,
    image: np.ndarray,
    title: str,
    cmap: Optional[str] = None,
    binary: bool = False,
    contour: Optional[np.ndarray] = None,
) -> None:
    """
    Add one image panel.

    binary=True forces fixed vmin/vmax so constant masks display correctly.
    contour is used to draw label boundaries over RGB/overlay panels.
    """
    ax = axes[panel_idx]

    if binary:
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(image, cmap=cmap)

    if contour is not None and np.any(contour > 0.5):
        ax.contour(
            contour,
            levels=[0.5],
            colors=["yellow"],
            linewidths=0.8,
        )

    ax.set_title(title, fontsize=9)
    ax.axis("off")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_filename(text: str, max_len: int = 150) -> str:
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


def visualize_sample(
    sample: Dict[str, Any],
    output_path: Path,
    dpi: int,
    sample_index: int,
) -> Dict[str, Any]:
    ensure_dir(output_path.parent)

    label = tensor_to_numpy(sample["label"])[0]
    valid = tensor_to_numpy(sample["valid_mask"])[0] > 0.5

    panels: List[Dict[str, Any]] = []

    rgb = None

    if "s2" in sample:
        s2 = tensor_to_numpy(sample["s2"])

        if s2.shape[0] >= 8:
            rgb = make_rgb(s2, bands=[3, 2, 1], valid_mask=valid)
            false_color = make_rgb(s2, bands=[7, 3, 2], valid_mask=valid)

            panels.append(
                {
                    "image": rgb,
                    "title": "S2 RGB B04/B03/B02",
                    "cmap": None,
                    "binary": False,
                    "contour": None,
                }
            )
            panels.append(
                {
                    "image": false_color,
                    "title": "S2 False Color B08/B04/B03",
                    "cmap": None,
                    "binary": False,
                    "contour": None,
                }
            )
        else:
            panels.append(
                {
                    "image": robust_stretch(s2[0], valid_mask=valid),
                    "title": "S2 band 1",
                    "cmap": "gray",
                    "binary": False,
                    "contour": None,
                }
            )

    if "s1" in sample:
        s1 = tensor_to_numpy(sample["s1"])

        if s1.shape[0] >= 1:
            panels.append(
                {
                    "image": robust_stretch(s1[0], valid_mask=valid),
                    "title": "S1 VV_dB",
                    "cmap": "gray",
                    "binary": False,
                    "contour": None,
                }
            )

        if s1.shape[0] >= 2:
            panels.append(
                {
                    "image": robust_stretch(s1[1], valid_mask=valid),
                    "title": "S1 VH_dB",
                    "cmap": "gray",
                    "binary": False,
                    "contour": None,
                }
            )

        if s1.shape[0] >= 3:
            panels.append(
                {
                    "image": robust_stretch(s1[2], valid_mask=valid),
                    "title": "S1 VV-minus-VH_dB",
                    "cmap": "gray",
                    "binary": False,
                    "contour": None,
                }
            )

    panels.append(
        {
            "image": label,
            "title": "Label mask",
            "cmap": "gray",
            "binary": True,
            "contour": None,
        }
    )

    if rgb is not None:
        overlay = make_filled_label_overlay(rgb, label)

        panels.append(
            {
                "image": overlay,
                "title": "Label overlay on S2 RGB",
                "cmap": None,
                "binary": False,
                "contour": label,
            }
        )

    panels.append(
        {
            "image": valid.astype("float32"),
            "title": "Valid mask",
            "cmap": "gray",
            "binary": True,
            "contour": None,
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
        )

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    patch_id = str(sample["patch_id"])
    city = str(sample["city"])
    split = str(sample["split"])
    region = str(sample.get("region", ""))
    pos_percent = safe_float(sample.get("label_positive_percent", math.nan))
    positive_pixels = int(np.sum(label > 0.5))
    valid_percent = 100.0 * float(np.mean(valid))

    title = (
        f"sample={sample_index} | city={city} | split={split} | region={region}\n"
        f"positive_pixels={positive_pixels} | label_positive_percent={pos_percent:.4f} | "
        f"valid={valid_percent:.2f}%\n"
        f"{patch_id}"
    )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return {
        "sample_index": sample_index,
        "patch_id": patch_id,
        "city": city,
        "split": split,
        "region": region,
        "positive_pixels": positive_pixels,
        "label_positive_percent": pos_percent,
        "valid_percent": valid_percent,
        "output_path": str(output_path),
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))

    if args.filter_set is None:
        filter_set_path = default_filter_set_path(
            output_root=output_root,
            split_strategy=args.split_strategy,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            filter_set_id=args.filter_set_id,
        )
    else:
        filter_set_path = args.filter_set

    if args.normalization == "none":
        normalization_json_path = None
    elif args.normalization_json is None:
        normalization_json_path = default_normalization_json_path(
            output_root=output_root,
            split_strategy=args.split_strategy,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            filter_set_id=args.filter_set_id,
        )
    else:
        normalization_json_path = args.normalization_json

    if args.output_dir is None:
        out_dir = default_output_dir(
            output_root=output_root,
            split_strategy=args.split_strategy,
            patch_size=args.patch_size,
            stride=args.stride,
            edge_mode=args.edge_mode,
            filter_set_id=args.filter_set_id,
            split=args.split,
            sample_mode=args.sample_mode,
        )
    else:
        out_dir = args.output_dir

    ensure_dir(out_dir)

    dataset_module = load_dataset_module()
    DatasetClass = dataset_module.BrazilFavelaGeoTiffDataset

    print("[INFO] Visualize PyTorch GeoTIFF dataset samples")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Filter set: {filter_set_path}")
    print(f"[INFO] Normalization JSON: {normalization_json_path}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Sample mode: {args.sample_mode}")
    print(f"[INFO] Num samples: {args.num_samples}")
    print(f"[INFO] Normalization: {args.normalization}")
    print(f"[INFO] Output directory: {out_dir}")

    dataset = DatasetClass(
        patch_csv=filter_set_path,
        split=args.split,
        modality="s2_s1",
        normalization_json=normalization_json_path,
        normalization=args.normalization,
        s2_band_set="all",
        s1_channel_set="all",
        max_patches=args.max_patches,
        treat_all_zero_s2_as_nodata=True,
    )

    print(f"[INFO] Dataset length: {len(dataset)}")

    selected_indices = select_indices(
        df=dataset.df,
        mode=args.sample_mode,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    print(f"[INFO] Selected indices: {selected_indices}")

    rows: List[Dict[str, Any]] = []

    for output_i, dataset_idx in enumerate(selected_indices):
        sample = dataset[dataset_idx]

        city = str(sample["city"])
        split = str(sample["split"])
        pos_percent = safe_float(sample.get("label_positive_percent", math.nan))

        file_name = safe_filename(
            f"sample_{output_i:03d}__idx_{dataset_idx:05d}__{city}__{split}__pos_{pos_percent:.4f}.png"
        )

        output_path = out_dir / file_name

        row = visualize_sample(
            sample=sample,
            output_path=output_path,
            dpi=args.dpi,
            sample_index=dataset_idx,
        )

        row["dataset_index"] = dataset_idx
        rows.append(row)

        print(f"[INFO] Wrote: {output_path}")

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "visualized_samples_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"[INFO] Wrote summary CSV: {summary_csv}")
    print("[INFO] Visualization completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())