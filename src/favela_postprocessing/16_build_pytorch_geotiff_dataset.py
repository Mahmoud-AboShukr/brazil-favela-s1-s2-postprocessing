#!/usr/bin/env python3
"""
PyTorch GeoTIFF Dataset for the Brazil favela segmentation dataset.

Purpose
-------
This script provides a reusable PyTorch Dataset class that reads patches directly
from GeoTIFF source rasters using the patch filter-set CSV.

It does NOT export H5 files and does NOT duplicate raster arrays.

It reads:
    - Sentinel-2 reflectance patches
    - SNAP Sentinel-1 patches
    - binary favela label patches

using:
    - patch filter-set CSV from 14_build_patch_filter_sets.py
    - training-only normalization JSON from 15_compute_normalization_stats.py

Default input
-------------
Patch list:

    <output_root>/metadata/patch_filter_sets_train_covered_region_test_ps512_st512_cover/
        filter_set_F02_quality_pass.csv

Normalization:

    <output_root>/metadata/
        normalization_stats_train_covered_region_test_ps512_st512_cover_F02_quality_pass.json

Example smoke test
------------------
    python3 src/favela_postprocessing/16_build_pytorch_geotiff_dataset.py \
        --config configs/default.yaml \
        --split train \
        --modality s2_s1 \
        --normalization standard \
        --num-samples 4 \
        --batch-size 2

Use S2 only:

    python3 src/favela_postprocessing/16_build_pytorch_geotiff_dataset.py \
        --config configs/default.yaml \
        --split train \
        --modality s2

Use S1 only:

    python3 src/favela_postprocessing/16_build_pytorch_geotiff_dataset.py \
        --config configs/default.yaml \
        --split train \
        --modality s1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import yaml


SCRIPT_NAME = "16_build_pytorch_geotiff_dataset.py"

DEFAULT_SPLIT_STRATEGY = "train_covered_region_test"
DEFAULT_PATCH_SIZE = 512
DEFAULT_STRIDE = 512
DEFAULT_EDGE_MODE = "cover"
DEFAULT_FILTER_SET_ID = "F02_quality_pass"

S2_BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]

S1_CHANNEL_NAMES = [
    "VV_dB",
    "VH_dB",
    "VV_minus_VH_dB",
]

S2_BAND_SETS = {
    "all": S2_BAND_NAMES,
    "rgb": ["B04", "B03", "B02"],
    "rgb_nir": ["B04", "B03", "B02", "B08"],
}

S1_CHANNEL_SETS = {
    "all": S1_CHANNEL_NAMES,
    "vv_vh": ["VV_dB", "VH_dB"],
    "vv": ["VV_dB"],
    "vh": ["VH_dB"],
    "vvdiff": ["VV_minus_VH_dB"],
}


def import_torch():
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for this script. Install it in your environment first. "
            f"Original import error: {exc}"
        )

    return torch, Dataset, DataLoader


torch, TorchDataset, DataLoader = import_torch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test a PyTorch Dataset that reads patches directly from GeoTIFFs."
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
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Dataset split to load. Default: train",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="s2_s1",
        choices=["s2", "s1", "s2_s1"],
        help="Input modality to return. Default: s2_s1",
    )
    parser.add_argument(
        "--s2-band-set",
        type=str,
        default="all",
        choices=sorted(S2_BAND_SETS.keys()),
        help="Sentinel-2 band subset. Default: all",
    )
    parser.add_argument(
        "--s1-channel-set",
        type=str,
        default="all",
        choices=sorted(S1_CHANNEL_SETS.keys()),
        help="Sentinel-1 channel subset. Default: all",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="standard",
        choices=["none", "standard", "clip_standard"],
        help=(
            "Normalization mode. "
            "'standard' uses (x - mean) / std. "
            "'clip_standard' clips to p2-p98 before standardization. "
            "Default: standard."
        ),
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Optional limit for smoke testing.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of individual samples to inspect. Default: 4",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for DataLoader smoke test. Default: 2",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 for debugging. Default: 0",
    )
    parser.add_argument(
        "--treat-all-zero-s2-as-nodata",
        action="store_true",
        default=True,
        help="Treat S2 pixels where all selected/full bands are zero as nodata. Default: true.",
    )
    parser.add_argument(
        "--do-not-treat-all-zero-s2-as-nodata",
        dest="treat_all_zero_s2_as_nodata",
        action="store_false",
        help="Disable treating all-zero S2 pixels as nodata.",
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


def normalize_city_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


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


def band_indices_from_names(all_names: Sequence[str], selected_names: Sequence[str]) -> List[int]:
    indices = []

    for name in selected_names:
        if name not in all_names:
            raise ValueError(f"Requested band/channel '{name}' is not in {all_names}")
        indices.append(all_names.index(name))

    return indices


def load_normalization_json(path: Optional[Path], normalization_mode: str) -> Optional[Dict[str, Any]]:
    if normalization_mode == "none":
        return None

    if path is None:
        raise ValueError("normalization_json path is required unless normalization='none'.")

    if not path.exists():
        raise FileNotFoundError(f"Normalization JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_patch_table(
    path: Path,
    split: str,
    max_patches: Optional[int],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Filter-set CSV not found: {path}")

    df = pd.read_csv(path)

    required = [
        "patch_id",
        "city",
        "split",
        "source_s2_path",
        "source_s1_path",
        "source_label_path",
        "row_start",
        "col_start",
        "patch_height",
        "patch_width",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Patch filter-set CSV is missing required columns: {missing}")

    df = df.copy()
    df["city"] = df["city"].map(normalize_city_name)

    if split != "all":
        df = df[df["split"].astype(str) == split].copy()

    if max_patches is not None:
        df = df.head(max_patches).copy()

    if df.empty:
        raise RuntimeError(f"No patches selected from {path} for split={split}")

    return df.reset_index(drop=True)


def window_from_row(row: pd.Series) -> Window:
    return Window(
        col_off=int(row["col_start"]),
        row_off=int(row["row_start"]),
        width=int(row["patch_width"]),
        height=int(row["patch_height"]),
    )


def valid_mask_for_array(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """
    Return a per-pixel validity mask.

    For [C, H, W], a pixel is valid only if all channels are finite and not nodata.
    """
    if arr.ndim == 2:
        valid = np.isfinite(arr)
        if nodata is not None and np.isfinite(nodata):
            valid &= arr != nodata
        return valid

    if arr.ndim == 3:
        valid = np.all(np.isfinite(arr), axis=0)
        if nodata is not None and np.isfinite(nodata):
            valid &= np.all(arr != nodata, axis=0)
        return valid

    raise ValueError(f"Unsupported array ndim: {arr.ndim}")


def all_zero_mask(arr: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    if arr.ndim == 2:
        return np.abs(arr) <= tolerance

    if arr.ndim == 3:
        return np.all(np.abs(arr) <= tolerance, axis=0)

    raise ValueError(f"Unsupported array ndim: {arr.ndim}")


def pad_array_chw(arr: np.ndarray, target_h: int, target_w: int, fill_value: float = 0.0) -> np.ndarray:
    """
    Pad [C, H, W] array to target size.

    Current tiling uses full-size cover patches, so this is mainly defensive.
    """
    c, h, w = arr.shape

    if h == target_h and w == target_w:
        return arr

    out = np.full((c, target_h, target_w), fill_value, dtype=arr.dtype)
    out[:, :h, :w] = arr
    return out


def pad_array_hw(arr: np.ndarray, target_h: int, target_w: int, fill_value: float = 0.0) -> np.ndarray:
    h, w = arr.shape

    if h == target_h and w == target_w:
        return arr

    out = np.full((target_h, target_w), fill_value, dtype=arr.dtype)
    out[:h, :w] = arr
    return out


def get_stats_for_names(
    norm: Optional[Dict[str, Any]],
    modality_key: str,
    names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    if norm is None:
        return {}

    if modality_key not in norm:
        raise KeyError(f"Normalization JSON missing modality key: {modality_key}")

    output = {}

    for name in names:
        if name not in norm[modality_key]:
            raise KeyError(f"Normalization JSON missing {modality_key}.{name}")

        item = norm[modality_key][name]

        output[name] = {
            "mean": float(item["mean"]),
            "std": float(item["std"]),
            "p2": float(item["p2"]),
            "p98": float(item["p98"]),
        }

    return output


def fill_invalid_with_mean(
    arr: np.ndarray,
    valid_mask: np.ndarray,
    names: Sequence[str],
    stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    out = arr.copy()

    invalid = ~valid_mask

    for c, name in enumerate(names):
        fill_value = stats.get(name, {}).get("mean", 0.0)
        out[c, invalid] = fill_value

    return out


def apply_normalization(
    arr: np.ndarray,
    names: Sequence[str],
    stats: Dict[str, Dict[str, float]],
    mode: str,
    eps: float = 1e-6,
) -> np.ndarray:
    if mode == "none":
        return arr.astype("float32", copy=False)

    out = arr.astype("float32", copy=True)

    for c, name in enumerate(names):
        item = stats[name]
        mean = float(item["mean"])
        std = float(item["std"])

        if mode == "clip_standard":
            p2 = float(item["p2"])
            p98 = float(item["p98"])
            out[c] = np.clip(out[c], p2, p98)

        if std < eps:
            std = 1.0

        out[c] = (out[c] - mean) / std

    return out.astype("float32", copy=False)


class BrazilFavelaGeoTiffDataset(TorchDataset):
    """
    PyTorch Dataset that reads favela segmentation patches directly from GeoTIFFs.

    Returned sample keys
    --------------------
    Depending on modality:
        sample["s2"]    FloatTensor [C_s2, H, W]
        sample["s1"]    FloatTensor [C_s1, H, W]

    Always:
        sample["label"] FloatTensor [1, H, W]
        sample["valid_mask"] FloatTensor [1, H, W]
        sample["patch_id"] str
        sample["city"] str
        sample["split"] str
    """

    def __init__(
        self,
        patch_csv: Path,
        split: str = "train",
        modality: str = "s2_s1",
        normalization_json: Optional[Path] = None,
        normalization: str = "standard",
        s2_band_set: str = "all",
        s1_channel_set: str = "all",
        max_patches: Optional[int] = None,
        treat_all_zero_s2_as_nodata: bool = True,
    ) -> None:
        super().__init__()

        if modality not in {"s2", "s1", "s2_s1"}:
            raise ValueError(f"Unsupported modality: {modality}")

        if normalization not in {"none", "standard", "clip_standard"}:
            raise ValueError(f"Unsupported normalization mode: {normalization}")

        if s2_band_set not in S2_BAND_SETS:
            raise ValueError(f"Unsupported s2_band_set: {s2_band_set}")

        if s1_channel_set not in S1_CHANNEL_SETS:
            raise ValueError(f"Unsupported s1_channel_set: {s1_channel_set}")

        self.patch_csv = Path(patch_csv)
        self.split = split
        self.modality = modality
        self.normalization = normalization
        self.treat_all_zero_s2_as_nodata = treat_all_zero_s2_as_nodata

        self.df = load_patch_table(
            path=self.patch_csv,
            split=split,
            max_patches=max_patches,
        )

        self.s2_names = S2_BAND_SETS[s2_band_set]
        self.s1_names = S1_CHANNEL_SETS[s1_channel_set]

        self.s2_indices = band_indices_from_names(S2_BAND_NAMES, self.s2_names)
        self.s1_indices = band_indices_from_names(S1_CHANNEL_NAMES, self.s1_names)

        self.norm = load_normalization_json(normalization_json, normalization)

        self.s2_stats = get_stats_for_names(
            self.norm,
            "s2_reflectance",
            self.s2_names,
        ) if normalization != "none" else {}

        self.s1_stats = get_stats_for_names(
            self.norm,
            "s1_snap",
            self.s1_names,
        ) if normalization != "none" else {}

    def __len__(self) -> int:
        return len(self.df)

    def _read_s2(self, row: pd.Series, window: Window) -> Tuple[np.ndarray, np.ndarray]:
        path = Path(str(row["source_s2_path"]))

        with rasterio.open(path) as src:
            full_arr = src.read(window=window).astype("float32", copy=False)

            valid = valid_mask_for_array(full_arr, src.nodata)

            if self.treat_all_zero_s2_as_nodata:
                valid &= ~all_zero_mask(full_arr)

            arr = full_arr[self.s2_indices, :, :]

        if self.normalization != "none":
            arr = fill_invalid_with_mean(arr, valid, self.s2_names, self.s2_stats)
            arr = apply_normalization(arr, self.s2_names, self.s2_stats, self.normalization)
        else:
            arr = arr.astype("float32", copy=False)
            arr[:, ~valid] = 0.0

        return arr, valid

    def _read_s1(self, row: pd.Series, window: Window) -> Tuple[np.ndarray, np.ndarray]:
        path = Path(str(row["source_s1_path"]))

        with rasterio.open(path) as src:
            full_arr = src.read(window=window).astype("float32", copy=False)
            valid = valid_mask_for_array(full_arr, src.nodata)
            arr = full_arr[self.s1_indices, :, :]

        if self.normalization != "none":
            arr = fill_invalid_with_mean(arr, valid, self.s1_names, self.s1_stats)
            arr = apply_normalization(arr, self.s1_names, self.s1_stats, self.normalization)
        else:
            arr = arr.astype("float32", copy=False)
            arr[:, ~valid] = 0.0

        return arr, valid

    def _read_label(self, row: pd.Series, window: Window) -> Tuple[np.ndarray, np.ndarray]:
        path = Path(str(row["source_label_path"]))

        with rasterio.open(path) as src:
            arr = src.read(1, window=window)

        finite = np.isfinite(arr)

        # Final labels are binary:
        #   0 = background
        #   1 = favela
        # Therefore 0 must be treated as a valid class, not nodata.
        valid = finite & ((arr == 0) | (arr == 1))
        label = (arr == 1).astype("float32")

        label[~valid] = 0.0

        return label[None, :, :], valid

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        window = window_from_row(row)

        target_h = int(row["patch_height"])
        target_w = int(row["patch_width"])

        sample: Dict[str, Any] = {}

        valid_masks = []

        if self.modality in {"s2", "s2_s1"}:
            s2, s2_valid = self._read_s2(row, window)
            s2 = pad_array_chw(s2, target_h, target_w, fill_value=0.0)
            s2_valid = pad_array_hw(s2_valid.astype("float32"), target_h, target_w, fill_value=0.0).astype(bool)

            sample["s2"] = torch.from_numpy(s2)
            valid_masks.append(s2_valid)

        if self.modality in {"s1", "s2_s1"}:
            s1, s1_valid = self._read_s1(row, window)
            s1 = pad_array_chw(s1, target_h, target_w, fill_value=0.0)
            s1_valid = pad_array_hw(s1_valid.astype("float32"), target_h, target_w, fill_value=0.0).astype(bool)

            sample["s1"] = torch.from_numpy(s1)
            valid_masks.append(s1_valid)

        label, label_valid = self._read_label(row, window)
        label = pad_array_chw(label, target_h, target_w, fill_value=0.0)
        label_valid = pad_array_hw(label_valid.astype("float32"), target_h, target_w, fill_value=0.0).astype(bool)

        valid_masks.append(label_valid)

        combined_valid = np.logical_and.reduce(valid_masks).astype("float32")[None, :, :]

        sample["label"] = torch.from_numpy(label.astype("float32"))
        sample["valid_mask"] = torch.from_numpy(combined_valid)

        sample["patch_id"] = str(row["patch_id"])
        sample["city"] = str(row["city"])
        sample["split"] = str(row["split"])
        sample["region"] = str(row["region"]) if "region" in row else ""
        sample["row_start"] = int(row["row_start"])
        sample["col_start"] = int(row["col_start"])

        if "label_positive_percent" in row:
            sample["label_positive_percent"] = float(row["label_positive_percent"])

        return sample


def describe_sample(sample: Dict[str, Any], prefix: str = "") -> None:
    print(f"{prefix}patch_id: {sample['patch_id']}")
    print(f"{prefix}city: {sample['city']}")
    print(f"{prefix}split: {sample['split']}")
    print(f"{prefix}region: {sample.get('region', '')}")

    if "s2" in sample:
        print(f"{prefix}s2 shape: {tuple(sample['s2'].shape)} dtype={sample['s2'].dtype}")
        print(
            f"{prefix}s2 min/max/mean: "
            f"{float(sample['s2'].min()):.4f} / "
            f"{float(sample['s2'].max()):.4f} / "
            f"{float(sample['s2'].mean()):.4f}"
        )

    if "s1" in sample:
        print(f"{prefix}s1 shape: {tuple(sample['s1'].shape)} dtype={sample['s1'].dtype}")
        print(
            f"{prefix}s1 min/max/mean: "
            f"{float(sample['s1'].min()):.4f} / "
            f"{float(sample['s1'].max()):.4f} / "
            f"{float(sample['s1'].mean()):.4f}"
        )

    print(f"{prefix}label shape: {tuple(sample['label'].shape)} dtype={sample['label'].dtype}")
    print(f"{prefix}label positive pixels: {int(sample['label'].sum().item())}")
    print(f"{prefix}valid mask percent: {100.0 * float(sample['valid_mask'].mean()):.2f}%")


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

    if args.normalization_json is None:
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

    if args.normalization == "none":
        normalization_json_path = None

    print("[INFO] PyTorch GeoTIFF Dataset smoke test")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Filter set: {filter_set_path}")
    print(f"[INFO] Normalization JSON: {normalization_json_path}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Modality: {args.modality}")
    print(f"[INFO] S2 band set: {args.s2_band_set}")
    print(f"[INFO] S1 channel set: {args.s1_channel_set}")
    print(f"[INFO] Normalization: {args.normalization}")
    print(f"[INFO] Max patches: {args.max_patches}")

    dataset = BrazilFavelaGeoTiffDataset(
        patch_csv=filter_set_path,
        split=args.split,
        modality=args.modality,
        normalization_json=normalization_json_path,
        normalization=args.normalization,
        s2_band_set=args.s2_band_set,
        s1_channel_set=args.s1_channel_set,
        max_patches=args.max_patches,
        treat_all_zero_s2_as_nodata=args.treat_all_zero_s2_as_nodata,
    )

    print(f"[INFO] Dataset length: {len(dataset)}")

    n = min(args.num_samples, len(dataset))

    for idx in range(n):
        print("=" * 100)
        print(f"[INFO] Inspecting sample {idx}")
        sample = dataset[idx]
        describe_sample(sample, prefix="       ")

    print("=" * 100)
    print("[INFO] DataLoader smoke test")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    batch = next(iter(loader))

    print(f"[INFO] Batch keys: {list(batch.keys())}")

    if "s2" in batch:
        print(f"[INFO] Batch s2 shape: {tuple(batch['s2'].shape)}")

    if "s1" in batch:
        print(f"[INFO] Batch s1 shape: {tuple(batch['s1'].shape)}")

    print(f"[INFO] Batch label shape: {tuple(batch['label'].shape)}")
    print(f"[INFO] Batch valid_mask shape: {tuple(batch['valid_mask'].shape)}")

    print("[INFO] PyTorch GeoTIFF Dataset smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())