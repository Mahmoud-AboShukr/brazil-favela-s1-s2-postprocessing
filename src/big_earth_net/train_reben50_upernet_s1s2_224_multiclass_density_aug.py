#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 6A:
reBEN ResNet50 partial layer4 fine-tune + UPerNet
Multiclass patch-density favela segmentation.

Classes:
    0 = background / non-favela
    1 = tiny favela pixels   : 0% < patch GT favela <= 1%
    2 = small favela pixels  : 1% < patch GT favela <= 5%
    3 = medium favela pixels : 5% < patch GT favela <= 20%
    4 = large favela pixels  : patch GT favela > 20%

Important:
    This is patch-density-aware segmentation, not true object-size segmentation.
    Favela pixels receive a class according to the total favela density of the patch.

Outputs:
    - checkpoints/best.pt
    - checkpoints/latest.pt
    - history.csv
    - summary.json
    - class_metrics_val.csv / class_metrics_test.csv
    - density_metrics_val.csv / density_metrics_test.csv
    - training curves
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    import rasterio
    from rasterio.windows import Window
except Exception as exc:
    raise RuntimeError(
        "This script requires rasterio. Install it in your venv first."
    ) from exc

try:
    from torchvision.models import resnet50
except Exception as exc:
    raise RuntimeError(
        "This script requires torchvision. Install torchvision in your venv first."
    ) from exc


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

CLASS_NAMES = ["background", "tiny", "small", "medium", "large"]
NUM_CLASSES = 5

DEFAULT_S1_BAND_INDICES = [1, 2]
DEFAULT_S2_BAND_INDICES_REBEN = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

DEFAULT_CLASS_WEIGHTS = [0.20, 3.00, 2.50, 1.50, 1.00]

DEFAULT_AUGMENTATION_MULTIPLIERS = {
    0: 1,  # empty/background
    1: 6,  # tiny
    2: 4,  # small
    3: 3,  # medium
    4: 2,  # large
}

AUGMENTATION_LIBRARY = [
    "none",
    "rot90",
    "rot180",
    "rot270",
    "hflip",
    "vflip",
    "hflip_rot90",
    "vflip_rot90",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def is_abs_path(value: str) -> bool:
    value = str(value)
    return os.path.isabs(value) or re.match(r"^[A-Za-z]:[\\/]", value) is not None


def resolve_path(value: str, instance_root: Path) -> Path:
    p = Path(str(value))
    if is_abs_path(str(value)):
        return p
    return instance_root / p


def find_column(df: pd.DataFrame, candidates: Sequence[str], contains: Sequence[str] = ()) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]

    for c in df.columns:
        lc = c.lower()
        if all(token.lower() in lc for token in contains):
            return c

    return None


def parse_patch_id_for_window(patch_id: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Parse strings such as:
        sao_luis__r001008__c001751__ps224__st112
    """
    if not isinstance(patch_id, str):
        return None, None, None

    m = re.search(r"__r(\d+)__c(\d+)__ps(\d+)", patch_id)
    if not m:
        return None, None, None

    row = int(m.group(1))
    col = int(m.group(2))
    ps = int(m.group(3))
    return row, col, ps


def row_get_first(row: pd.Series, names: Sequence[str]) -> Optional[object]:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return None


def get_window_from_row(row: pd.Series, patch_size: int) -> Optional[Window]:
    row_candidates = ["row", "r", "row_off", "row_offset", "y", "yoff", "y_off"]
    col_candidates = ["col", "c", "col_off", "col_offset", "x", "xoff", "x_off"]

    r = row_get_first(row, row_candidates)
    c = row_get_first(row, col_candidates)

    if r is not None and c is not None:
        try:
            return Window(int(c), int(r), patch_size, patch_size)
        except Exception:
            pass

    patch_id_col = None
    for name in ["patch_id", "id", "patch", "sample_id"]:
        if name in row.index:
            patch_id_col = name
            break

    if patch_id_col is not None:
        rr, cc, ps = parse_patch_id_for_window(str(row[patch_id_col]))
        if rr is not None and cc is not None:
            return Window(cc, rr, patch_size, patch_size)

    return None


def pad_or_crop_chw(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Ensure array has shape C x out_h x out_w.
    """
    c, h, w = arr.shape
    result = np.zeros((c, out_h, out_w), dtype=arr.dtype)
    copy_h = min(h, out_h)
    copy_w = min(w, out_w)
    result[:, :copy_h, :copy_w] = arr[:, :copy_h, :copy_w]
    return result


def pad_or_crop_hw(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    result = np.zeros((out_h, out_w), dtype=arr.dtype)
    h, w = arr.shape
    copy_h = min(h, out_h)
    copy_w = min(w, out_w)
    result[:copy_h, :copy_w] = arr[:copy_h, :copy_w]
    return result


def choose_band_indices(src_count: int, preferred_indices: Sequence[int], expected_count: int) -> List[int]:
    """
    Rasterio band indices are 1-based.
    """
    if src_count >= max(preferred_indices):
        return list(preferred_indices)

    if src_count == expected_count:
        return list(range(1, expected_count + 1))

    if src_count > expected_count:
        return list(range(1, expected_count + 1))

    return list(range(1, src_count + 1))


def read_raster_chw(
    path: Path,
    preferred_band_indices: Sequence[int],
    expected_count: int,
    patch_size: int,
    window: Optional[Window] = None,
) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim == 3 and arr.shape[-1] <= 32 and arr.shape[0] > 32:
            arr = np.transpose(arr, (2, 0, 1))
        arr = arr.astype(np.float32)
        return pad_or_crop_chw(arr, patch_size, patch_size)

    with rasterio.open(path) as src:
        band_indices = choose_band_indices(src.count, preferred_band_indices, expected_count)

        if window is not None and (src.height > patch_size or src.width > patch_size):
            arr = src.read(band_indices, window=window, boundless=True, fill_value=0)
        else:
            arr = src.read(band_indices)

    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = pad_or_crop_chw(arr, patch_size, patch_size)

    if arr.shape[0] < expected_count:
        padded = np.zeros((expected_count, patch_size, patch_size), dtype=np.float32)
        padded[: arr.shape[0]] = arr
        arr = padded

    return arr


def read_mask_hw(path: Path, patch_size: int, window: Optional[Window] = None) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr[0] if arr.shape[0] <= arr.shape[-1] else arr[:, :, 0]
        arr = arr.astype(np.uint8)
        return pad_or_crop_hw(arr, patch_size, patch_size)

    with rasterio.open(path) as src:
        if window is not None and (src.height > patch_size or src.width > patch_size):
            arr = src.read(1, window=window, boundless=True, fill_value=0)
        else:
            arr = src.read(1)

    arr = np.nan_to_num(arr, nan=0.0)
    arr = (arr > 0).astype(np.uint8)
    return pad_or_crop_hw(arr, patch_size, patch_size)


def per_sample_channel_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-sample, per-channel standardisation.
    """
    x = x.astype(np.float32, copy=False)

    for c in range(x.shape[0]):
        band = x[c]
        valid = np.isfinite(band)
        if not valid.any():
            x[c] = 0.0
            continue

        vals = band[valid]
        mean = float(vals.mean())
        std = float(vals.std())

        if std < eps:
            x[c] = 0.0
        else:
            x[c] = (band - mean) / (std + eps)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def density_pct_to_class(gt_pos_pct: float) -> int:
    if gt_pos_pct <= 0.0:
        return 0
    if gt_pos_pct <= 1.0:
        return 1
    if gt_pos_pct <= 5.0:
        return 2
    if gt_pos_pct <= 20.0:
        return 3
    return 4


def density_class_to_name(cls: int) -> str:
    return CLASS_NAMES[int(cls)]


def binary_mask_to_multiclass(mask: np.ndarray, density_class: int) -> np.ndarray:
    target = np.zeros(mask.shape, dtype=np.int64)
    if density_class > 0:
        target[mask > 0] = int(density_class)
    return target


def apply_augmentation(x: np.ndarray, y: np.ndarray, aug: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: C,H,W
    y: H,W
    """
    if aug == "none":
        pass
    elif aug == "rot90":
        x = np.rot90(x, k=1, axes=(1, 2))
        y = np.rot90(y, k=1, axes=(0, 1))
    elif aug == "rot180":
        x = np.rot90(x, k=2, axes=(1, 2))
        y = np.rot90(y, k=2, axes=(0, 1))
    elif aug == "rot270":
        x = np.rot90(x, k=3, axes=(1, 2))
        y = np.rot90(y, k=3, axes=(0, 1))
    elif aug == "hflip":
        x = np.flip(x, axis=2)
        y = np.flip(y, axis=1)
    elif aug == "vflip":
        x = np.flip(x, axis=1)
        y = np.flip(y, axis=0)
    elif aug == "hflip_rot90":
        x = np.flip(x, axis=2)
        y = np.flip(y, axis=1)
        x = np.rot90(x, k=1, axes=(1, 2))
        y = np.rot90(y, k=1, axes=(0, 1))
    elif aug == "vflip_rot90":
        x = np.flip(x, axis=1)
        y = np.flip(y, axis=0)
        x = np.rot90(x, k=1, axes=(1, 2))
        y = np.rot90(y, k=1, axes=(0, 1))
    else:
        raise ValueError(f"Unknown augmentation: {aug}")

    return np.ascontiguousarray(x), np.ascontiguousarray(y)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

@dataclass
class DatasetColumns:
    s1_col: str
    s2_col: str
    label_col: str
    city_col: Optional[str]
    region_col: Optional[str]
    patch_id_col: Optional[str]
    gt_pct_col: Optional[str]


def infer_columns(df: pd.DataFrame) -> DatasetColumns:
    s1_col = find_column(
        df,
        candidates=[
            "s1_path", "s1_file", "s1", "s1_patch", "s1_grd_path",
            "s1_snap_path", "s1_vvvh_path", "sar_path", "sar_file"
        ],
        contains=("s1", "path"),
    )

    s2_col = find_column(
        df,
        candidates=[
            "s2_path", "s2_file", "s2", "s2_patch", "s2_bands_path",
            "optical_path", "optical_file"
        ],
        contains=("s2", "path"),
    )

    label_col = find_column(
        df,
        candidates=[
            "label_path", "mask_path", "target_path", "label_file",
            "mask_file", "y_path", "binary_label_path", "favela_mask_path"
        ],
        contains=("label", "path"),
    )

    if label_col is None:
        label_col = find_column(df, candidates=[], contains=("mask", "path"))

    city_col = find_column(df, candidates=["city", "city_name", "municipality"], contains=("city",))
    region_col = find_column(df, candidates=["region", "region_name"], contains=("region",))
    patch_id_col = find_column(df, candidates=["patch_id", "id", "patch", "sample_id"], contains=("patch",))
    gt_pct_col = find_column(
        df,
        candidates=[
            "gt_pos_pct", "favela_pct", "label_pos_pct", "gt_favela_pct",
            "favela_pixel_pct", "gt_density_pct"
        ],
        contains=("pct",),
    )

    missing = []
    if s1_col is None:
        missing.append("S1 path column")
    if s2_col is None:
        missing.append("S2 path column")
    if label_col is None:
        missing.append("label/mask path column")

    if missing:
        raise ValueError(
            "Could not infer required columns: "
            + ", ".join(missing)
            + "\nAvailable columns are:\n"
            + "\n".join([f"  - {c}" for c in df.columns])
            + "\n\nFix by renaming the CSV columns or adding explicit support in infer_columns()."
        )

    return DatasetColumns(
        s1_col=s1_col,
        s2_col=s2_col,
        label_col=label_col,
        city_col=city_col,
        region_col=region_col,
        patch_id_col=patch_id_col,
        gt_pct_col=gt_pct_col,
    )


class FavelaDensityMulticlassDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        columns: DatasetColumns,
        instance_root: Path,
        patch_size: int,
        normalize: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.columns = columns
        self.instance_root = instance_root
        self.patch_size = patch_size
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.df)

    def _get_density_class(self, row: pd.Series, binary_mask: np.ndarray) -> Tuple[float, int]:
        if self.columns.gt_pct_col is not None and self.columns.gt_pct_col in row.index and pd.notna(row[self.columns.gt_pct_col]):
            try:
                gt_pct = float(row[self.columns.gt_pct_col])
            except Exception:
                gt_pct = float(binary_mask.mean() * 100.0)
        else:
            gt_pct = float(binary_mask.mean() * 100.0)

        cls = density_pct_to_class(gt_pct)
        return gt_pct, cls

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        s1_path = resolve_path(str(row[self.columns.s1_col]), self.instance_root)
        s2_path = resolve_path(str(row[self.columns.s2_col]), self.instance_root)
        label_path = resolve_path(str(row[self.columns.label_col]), self.instance_root)

        window = get_window_from_row(row, self.patch_size)

        s1 = read_raster_chw(
            path=s1_path,
            preferred_band_indices=DEFAULT_S1_BAND_INDICES,
            expected_count=2,
            patch_size=self.patch_size,
            window=window,
        )

        s2 = read_raster_chw(
            path=s2_path,
            preferred_band_indices=DEFAULT_S2_BAND_INDICES_REBEN,
            expected_count=10,
            patch_size=self.patch_size,
            window=window,
        )

        binary_mask = read_mask_hw(label_path, patch_size=self.patch_size, window=window)

        gt_pct, density_cls = self._get_density_class(row, binary_mask)
        target = binary_mask_to_multiclass(binary_mask, density_cls)

        x = np.concatenate([s1, s2], axis=0).astype(np.float32)

        if self.normalize:
            x = per_sample_channel_normalize(x)

        aug = str(row["augmentation"]) if "augmentation" in row.index else "none"
        x, target = apply_augmentation(x, target, aug)

        patch_id = str(row[self.columns.patch_id_col]) if self.columns.patch_id_col else f"idx_{idx}"
        city = str(row[self.columns.city_col]) if self.columns.city_col else "unknown_city"
        region = str(row[self.columns.region_col]) if self.columns.region_col else "unknown_region"

        return {
            "image": torch.from_numpy(x).float(),
            "target": torch.from_numpy(target).long(),
            "density_class": torch.tensor(density_cls, dtype=torch.long),
            "gt_pos_pct": torch.tensor(gt_pct, dtype=torch.float32),
            "patch_id": patch_id,
            "city": city,
            "region": region,
        }


def compute_density_for_dataframe(
    df: pd.DataFrame,
    columns: DatasetColumns,
    instance_root: Path,
    patch_size: int,
) -> pd.DataFrame:
    """
    Compute GT favela percentage and density class for each row if not already present.
    """
    df = df.copy()
    gt_pcts = []
    density_classes = []

    print(f"Computing density classes for {len(df)} rows...")

    for i, row in df.iterrows():
        if columns.gt_pct_col is not None and columns.gt_pct_col in row.index and pd.notna(row[columns.gt_pct_col]):
            gt_pct = float(row[columns.gt_pct_col])
        else:
            label_path = resolve_path(str(row[columns.label_col]), instance_root)
            window = get_window_from_row(row, patch_size)
            mask = read_mask_hw(label_path, patch_size=patch_size, window=window)
            gt_pct = float(mask.mean() * 100.0)

        cls = density_pct_to_class(gt_pct)
        gt_pcts.append(gt_pct)
        density_classes.append(cls)

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(df)} rows")

    df["gt_pos_pct_computed"] = gt_pcts
    df["density_class"] = density_classes
    df["density_label"] = [density_class_to_name(c) for c in density_classes]
    return df


def build_density_augmented_train_manifest(
    train_df: pd.DataFrame,
    columns: DatasetColumns,
    instance_root: Path,
    patch_size: int,
    multipliers: Dict[int, int],
    max_aug_per_patch: int,
    output_csv: Path,
) -> pd.DataFrame:
    cached = output_csv

    if cached.exists():
        print(f"Loading cached augmented manifest: {cached}")
        return pd.read_csv(cached)

    train_df = compute_density_for_dataframe(
        train_df,
        columns=columns,
        instance_root=instance_root,
        patch_size=patch_size,
    )

    rows = []
    for _, row in train_df.iterrows():
        density_cls = int(row["density_class"])
        multiplier = int(multipliers.get(density_cls, 1))
        multiplier = max(1, min(multiplier, max_aug_per_patch))

        selected_augs = AUGMENTATION_LIBRARY[:multiplier]

        for aug in selected_augs:
            new_row = row.copy()
            new_row["augmentation"] = aug
            rows.append(new_row)

    aug_df = pd.DataFrame(rows)
    ensure_dir(output_csv.parent)
    aug_df.to_csv(output_csv, index=False)

    print("\nAugmented manifest summary:")
    print(f"  Original train rows: {len(train_df)}")
    print(f"  Augmented train rows: {len(aug_df)}")
    print("\nRows by density class after augmentation:")
    print(aug_df["density_label"].value_counts().to_string())

    print(f"\nSaved augmented manifest to: {output_csv}")

    return aug_df


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class ResNet50Encoder(nn.Module):
    def __init__(self, in_channels: int = 12):
        super().__init__()
        backbone = resnet50(weights=None)

        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        self.backbone = backbone
        self.out_channels = [256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b = self.backbone

        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)

        c1 = b.layer1(x)
        c2 = b.layer2(c1)
        c3 = b.layer3(c2)
        c4 = b.layer4(c3)

        return [c1, c2, c3, c4]


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_scales: Sequence[int] = (1, 2, 3, 6)):
        super().__init__()

        self.stages = nn.ModuleList()
        for scale in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pyramids = [x]

        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            pyramids.append(y)

        return self.bottleneck(torch.cat(pyramids, dim=1))


class UPerNetDecoder(nn.Module):
    def __init__(
        self,
        feature_channels: Sequence[int],
        decoder_channels: int,
        ppm_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        c1, c2, c3, c4 = feature_channels

        self.ppm = PyramidPoolingModule(
            in_channels=c4,
            out_channels=decoder_channels,
            pool_scales=(1, 2, 3, 6),
        )

        self.lateral3 = nn.Conv2d(c3, decoder_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(c2, decoder_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(c1, decoder_channels, kernel_size=1)

        self.fpn3 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn2 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn1 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * 4, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor], output_size: Tuple[int, int]) -> torch.Tensor:
        c1, c2, c3, c4 = features

        p4 = self.ppm(c4)

        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.fpn3(p3)

        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.fpn2(p2)

        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.fpn1(p1)

        size = p1.shape[-2:]
        p2_up = F.interpolate(p2, size=size, mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=size, mode="bilinear", align_corners=False)
        p4_up = F.interpolate(p4, size=size, mode="bilinear", align_corners=False)

        logits = self.fuse(torch.cat([p1, p2_up, p3_up, p4_up], dim=1))
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


class FavelaMulticlassSegModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        decoder_channels: int,
        ppm_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = ResNet50Encoder(in_channels=in_channels)
        self.decoder = UPerNetDecoder(
            feature_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            ppm_channels=ppm_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        features = self.encoder(x)
        logits = self.decoder(features, output_size=output_size)
        return logits


def load_reben_weights_if_available(model: FavelaMulticlassSegModel, repo_id: str) -> Dict[str, object]:
    """
    Tries to load compatible ResNet50 weights from Hugging Face.

    This is intentionally tolerant:
    - It downloads the repo snapshot.
    - It searches for .safetensors, .bin, .pth, .pt, .ckpt.
    - It loads only keys whose names and shapes match torchvision ResNet50.
    - If no compatible weights are found, training still runs, but the encoder is random.

    For your current project, this should work if the reBEN checkpoint stores ResNet-style keys.
    """
    info = {
        "repo_id": repo_id,
        "loaded": False,
        "checkpoint_path": None,
        "matched_keys": 0,
        "skipped_keys": 0,
        "message": "",
    }

    if not repo_id:
        info["message"] = "No repo_id provided; using random encoder initialisation."
        print(info["message"])
        return info

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        info["message"] = "huggingface_hub is not installed; using random encoder initialisation."
        print(info["message"])
        return info

    try:
        repo_dir = Path(snapshot_download(repo_id=repo_id))
    except Exception as exc:
        info["message"] = f"Could not download HF repo {repo_id}: {exc}"
        print(info["message"])
        return info

    candidates = []
    for ext in ["*.safetensors", "*.bin", "*.pth", "*.pt", "*.ckpt"]:
        candidates.extend(repo_dir.rglob(ext))

    if not candidates:
        info["message"] = f"No checkpoint files found in {repo_dir}; using random encoder initialisation."
        print(info["message"])
        return info

    ckpt_path = candidates[0]
    info["checkpoint_path"] = str(ckpt_path)

    try:
        if ckpt_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except Exception as exc:
                raise RuntimeError("safetensors is required to load .safetensors checkpoints.") from exc
            state = load_file(str(ckpt_path), device="cpu")
        else:
            state = torch.load(str(ckpt_path), map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]

        if not isinstance(state, dict):
            raise RuntimeError("Checkpoint does not contain a state dict.")

        backbone = model.encoder.backbone
        target_state = backbone.state_dict()
        load_state = {}

        def clean_key(k: str) -> str:
            prefixes = [
                "module.",
                "model.",
                "encoder.",
                "backbone.",
                "net.",
                "resnet.",
                "student.",
                "teacher.",
            ]
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if k.startswith(p):
                        k = k[len(p):]
                        changed = True
            return k

        matched = 0
        skipped = 0

        for k, v in state.items():
            kk = clean_key(k)

            if kk in target_state and tuple(v.shape) == tuple(target_state[kk].shape):
                load_state[kk] = v
                matched += 1
            else:
                skipped += 1

        missing, unexpected = backbone.load_state_dict(load_state, strict=False)

        info["matched_keys"] = matched
        info["skipped_keys"] = skipped
        info["loaded"] = matched > 0
        info["message"] = (
            f"Loaded compatible encoder weights from {ckpt_path}. "
            f"Matched keys: {matched}. Skipped keys: {skipped}. "
            f"Missing keys after partial load: {len(missing)}. Unexpected: {len(unexpected)}."
        )

        print(info["message"])
        return info

    except Exception as exc:
        info["message"] = f"Failed to load weights from {ckpt_path}: {exc}"
        print(info["message"])
        return info


def apply_freeze_mode(model: FavelaMulticlassSegModel, mode: str) -> Dict[str, int]:
    """
    mode:
        all    : freeze entire encoder
        layer4 : freeze encoder except layer4
        none   : fine-tune entire encoder
    """
    if mode not in {"all", "layer4", "none"}:
        raise ValueError(f"Invalid freeze mode: {mode}")

    for p in model.encoder.parameters():
        p.requires_grad = False

    if mode == "none":
        for p in model.encoder.parameters():
            p.requires_grad = True

    elif mode == "layer4":
        for p in model.encoder.backbone.layer4.parameters():
            p.requires_grad = True

    elif mode == "all":
        pass

    total_backbone = sum(p.numel() for p in model.encoder.parameters())
    trainable_backbone = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    frozen_backbone = total_backbone - trainable_backbone

    decoder_total = sum(p.numel() for p in model.decoder.parameters())
    decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

    print("\nFreeze summary:")
    print(f"  Freeze mode: {mode}")
    print(f"  Backbone params total:    {total_backbone:,}")
    print(f"  Backbone params trainable:{trainable_backbone:,}")
    print(f"  Backbone params frozen:   {frozen_backbone:,}")
    print(f"  Decoder params total:     {decoder_total:,}")
    print(f"  Decoder params trainable: {decoder_trainable:,}")

    return {
        "backbone_total": total_backbone,
        "backbone_trainable": trainable_backbone,
        "backbone_frozen": frozen_backbone,
        "decoder_total": decoder_total,
        "decoder_trainable": decoder_trainable,
    }


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

class MulticlassDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        include_background: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.eps = eps

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)

        target_oh = F.one_hot(target.clamp(0, self.num_classes - 1), num_classes=self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = torch.sum(probs * target_oh, dim=dims)
        denominator = torch.sum(probs + target_oh, dim=dims)

        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        loss_per_class = 1.0 - dice

        start = 0 if self.include_background else 1
        loss_per_class = loss_per_class[start:]

        if self.class_weights is not None:
            weights = self.class_weights[start:]
            weights = weights / (weights.sum() + self.eps)
            return torch.sum(loss_per_class * weights)

        return loss_per_class.mean()


class WeightedCEDiceLoss(nn.Module):
    def __init__(
        self,
        class_weights: Sequence[float],
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        include_background_in_dice: bool = False,
    ):
        super().__init__()
        weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("weights", weights)

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce = nn.CrossEntropyLoss(weight=self.weights)
        self.dice = MulticlassDiceLoss(
            num_classes=len(class_weights),
            class_weights=self.weights,
            include_background=include_background_in_dice,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        total = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        parts = {
            "ce_loss": float(ce_loss.detach().cpu()),
            "dice_loss": float(dice_loss.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }

        return total, parts


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def confusion_matrix_multiclass(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = pred.detach().view(-1).long()
    target = target.detach().view(-1).long()

    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]

    idx = target * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    cm = cm.reshape(num_classes, num_classes)
    return cm.cpu()


def metrics_from_confusion(cm: torch.Tensor) -> pd.DataFrame:
    cm = cm.double()
    total = cm.sum().item()

    rows = []
    for c, name in enumerate(CLASS_NAMES):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        tn = total - tp - fp - fn

        iou = tp / (tp + fp + fn + 1e-9)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        rows.append({
            "class_id": c,
            "class_name": name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "support_pixels": cm[c, :].sum().item(),
            "pred_pixels": cm[:, c].sum().item(),
        })

    return pd.DataFrame(rows)


def binary_stats_from_prediction(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred_bin = pred > 0
    target_bin = target > 0

    tp = torch.logical_and(pred_bin, target_bin).sum().item()
    fp = torch.logical_and(pred_bin, ~target_bin).sum().item()
    fn = torch.logical_and(~pred_bin, target_bin).sum().item()
    tn = torch.logical_and(~pred_bin, ~target_bin).sum().item()

    iou = tp / (tp + fp + fn + 1e-9)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-9)

    pred_pct = 100.0 * (tp + fp) / (tp + fp + fn + tn + 1e-9)
    gt_pct = 100.0 * (tp + fn) / (tp + fp + fn + tn + 1e-9)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "pred_pos_pct": pred_pct,
        "gt_pos_pct": gt_pct,
        "pred_minus_gt_pct": pred_pct - gt_pct,
    }


def aggregate_binary_stats(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    tp = sum(s["tp"] for s in stats_list)
    fp = sum(s["fp"] for s in stats_list)
    fn = sum(s["fn"] for s in stats_list)
    tn = sum(s["tn"] for s in stats_list)

    iou = tp / (tp + fp + fn + 1e-9)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-9)

    pred_pct = 100.0 * (tp + fp) / (tp + fp + fn + tn + 1e-9)
    gt_pct = 100.0 * (tp + fn) / (tp + fp + fn + tn + 1e-9)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "pred_pos_pct": pred_pct,
        "gt_pos_pct": gt_pct,
        "pred_minus_gt_pct": pred_pct - gt_pct,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: WeightedCEDiceLoss,
    device: torch.device,
    num_classes: int,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    model.eval()

    total_loss = 0.0
    n_batches = 0

    cm_total = torch.zeros((num_classes, num_classes), dtype=torch.long)

    binary_stats_all = []
    density_binary_stats = {c: [] for c in range(num_classes)}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        logits = model(images)
        loss, _ = loss_fn(logits, targets)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        cm_total += confusion_matrix_multiclass(preds.cpu(), targets.cpu(), num_classes=num_classes)

        for i in range(images.shape[0]):
            stats = binary_stats_from_prediction(preds[i].cpu(), targets[i].cpu())
            binary_stats_all.append(stats)

            dcls = int(batch["density_class"][i].item())
            density_binary_stats[dcls].append(stats)

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    class_df = metrics_from_confusion(cm_total)

    favela_classes = class_df[class_df["class_id"] > 0]
    mean_favela_iou = float(favela_classes["iou"].mean())
    mean_favela_dice = float(favela_classes["dice"].mean())

    binary_all = aggregate_binary_stats(binary_stats_all)

    density_rows = []
    for c in range(num_classes):
        if len(density_binary_stats[c]) == 0:
            continue

        agg = aggregate_binary_stats(density_binary_stats[c])
        row = {
            "density_class": c,
            "density_label": density_class_to_name(c),
            "n_patches": len(density_binary_stats[c]),
        }
        row.update({f"binary_{k}": v for k, v in agg.items()})
        density_rows.append(row)

    density_df = pd.DataFrame(density_rows)

    summary = {
        "loss": total_loss / max(1, n_batches),
        "binary_iou_favela": binary_all["iou"],
        "binary_dice_favela": binary_all["dice"],
        "binary_precision_favela": binary_all["precision"],
        "binary_recall_favela": binary_all["recall"],
        "binary_accuracy": binary_all["accuracy"],
        "binary_pred_pos_pct": binary_all["pred_pos_pct"],
        "binary_gt_pos_pct": binary_all["gt_pos_pct"],
        "binary_pred_minus_gt_pct": binary_all["pred_minus_gt_pct"],
        "mean_favela_class_iou": mean_favela_iou,
        "mean_favela_class_dice": mean_favela_dice,
    }

    return summary, class_df, density_df


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_history(history_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)

    def plot_metric(metric: str, filename: str, ylabel: str):
        plt.figure(figsize=(8, 5))
        if f"train_{metric}" in history_df.columns:
            plt.plot(history_df["epoch"], history_df[f"train_{metric}"], label=f"train {metric}")
        if f"val_{metric}" in history_df.columns:
            plt.plot(history_df["epoch"], history_df[f"val_{metric}"], label=f"val {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    plot_metric("loss", "curve_loss.png", "Loss")
    plot_metric("binary_iou_favela", "curve_binary_iou_favela.png", "Binary Favela IoU")
    plot_metric("binary_dice_favela", "curve_binary_dice_favela.png", "Binary Favela Dice")
    plot_metric("binary_precision_favela", "curve_binary_precision_favela.png", "Binary Favela Precision")
    plot_metric("binary_recall_favela", "curve_binary_recall_favela.png", "Binary Favela Recall")
    plot_metric("mean_favela_class_iou", "curve_mean_favela_class_iou.png", "Mean Favela-Class IoU")


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------

def make_optimizer(
    model: FavelaMulticlassSegModel,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": head_lr})

    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
    extra: Optional[dict] = None,
) -> None:
    ensure_dir(path.parent)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": best_metric,
        "args": vars(args),
        "class_names": CLASS_NAMES,
    }
    if extra:
        ckpt.update(extra)

    torch.save(ckpt, path)


def load_checkpoint_if_requested(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, float]:
    if not path.exists():
        print(f"No checkpoint found at {path}; starting from epoch 1.")
        return 1, -1.0

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])

    start_epoch = int(ckpt["epoch"]) + 1
    best_metric = float(ckpt.get("best_metric", -1.0))

    print(f"Resumed from {path}")
    print(f"  start_epoch={start_epoch}")
    print(f"  best_metric={best_metric:.6f}")

    return start_epoch, best_metric


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: WeightedCEDiceLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    amp: bool,
    accum_steps: int,
    max_train_batches: Optional[int],
    grad_clip_norm: Optional[float],
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if max_train_batches is not None and step >= max_train_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with autocast(enabled=amp):
            logits = model(images)
            loss, parts = loss_fn(logits, targets)
            loss_for_backward = loss / accum_steps

        scaler.scale(loss_for_backward).backward()

        if (step + 1) % accum_steps == 0:
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += parts["total_loss"]
        total_ce += parts["ce_loss"]
        total_dice += parts["dice_loss"]
        n_batches += 1

        if (step + 1) % 50 == 0:
            print(
                f"    batch {step + 1}/{len(loader)} "
                f"loss={total_loss / n_batches:.4f} "
                f"ce={total_ce / n_batches:.4f} "
                f"dice={total_dice / n_batches:.4f}"
            )

    if n_batches > 0 and n_batches % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss": total_loss / max(1, n_batches),
        "ce_loss": total_ce / max(1, n_batches),
        "dice_loss": total_dice / max(1, n_batches),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--instance-root", type=str, required=True)
    parser.add_argument("--split-dir", type=str, default=None)

    parser.add_argument("--run-name", type=str, default="exp06_reben50_layer4_multiclass_density_aug_wce_dice")
    parser.add_argument("--output-root", type=str, default=None)

    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--ppm-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--freeze-backbone-mode", type=str, default="layer4", choices=["all", "layer4", "none"])
    parser.add_argument("--reben-repo-id", type=str, default="BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0")

    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--class-weights", type=float, nargs=5, default=DEFAULT_CLASS_WEIGHTS)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)

    parser.add_argument("--empty-aug", type=int, default=1)
    parser.add_argument("--tiny-aug", type=int, default=6)
    parser.add_argument("--small-aug", type=int, default=4)
    parser.add_argument("--medium-aug", type=int, default=3)
    parser.add_argument("--large-aug", type=int, default=2)
    parser.add_argument("--max-aug-per-patch", type=int, default=8)

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--resume", type=str, default="none", choices=["none", "latest", "best"])
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)

    instance_root = Path(args.instance_root)

    if args.split_dir is None:
        split_dir = instance_root / "metadata" / "big_earth_net" / "same_city_low_holdout_ps224_st112_cover"
    else:
        split_dir = Path(args.split_dir)

    if args.output_root is None:
        output_root = (
            instance_root
            / "experiments"
            / "big_earth_net"
            / "reben50_upernet_s1s2_multiclass_density_ps224"
        )
    else:
        output_root = Path(args.output_root)

    run_dir = output_root / args.run_name

    if args.overwrite and run_dir.exists():
        print(f"Removing previous run directory: {run_dir}")
        shutil.rmtree(run_dir)

    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "metrics")
    ensure_dir(run_dir / "manifests")

    save_json(vars(args), run_dir / "config.json")

    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train CSV: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing val CSV: {val_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test CSV: {test_csv}")

    train_df_base = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print("\nLoaded splits:")
    print(f"  train: {len(train_df_base)} rows")
    print(f"  val:   {len(val_df)} rows")
    print(f"  test:  {len(test_df)} rows")

    columns = infer_columns(train_df_base)

    print("\nInferred columns:")
    print(f"  S1:       {columns.s1_col}")
    print(f"  S2:       {columns.s2_col}")
    print(f"  Label:    {columns.label_col}")
    print(f"  City:     {columns.city_col}")
    print(f"  Region:   {columns.region_col}")
    print(f"  Patch ID: {columns.patch_id_col}")
    print(f"  GT pct:   {columns.gt_pct_col}")

    multipliers = {
        0: args.empty_aug,
        1: args.tiny_aug,
        2: args.small_aug,
        3: args.medium_aug,
        4: args.large_aug,
    }

    train_manifest_csv = run_dir / "manifests" / "train_multiclass_density_augmented.csv"

    train_df = build_density_augmented_train_manifest(
        train_df=train_df_base,
        columns=columns,
        instance_root=instance_root,
        patch_size=args.patch_size,
        multipliers=multipliers,
        max_aug_per_patch=args.max_aug_per_patch,
        output_csv=train_manifest_csv,
    )

    val_df = val_df.copy()
    val_df["augmentation"] = "none"

    test_df = test_df.copy()
    test_df["augmentation"] = "none"

    train_ds = FavelaDensityMulticlassDataset(
        train_df,
        columns=columns,
        instance_root=instance_root,
        patch_size=args.patch_size,
        normalize=True,
    )

    val_ds = FavelaDensityMulticlassDataset(
        val_df,
        columns=columns,
        instance_root=instance_root,
        patch_size=args.patch_size,
        normalize=True,
    )

    test_ds = FavelaDensityMulticlassDataset(
        test_df,
        columns=columns,
        instance_root=instance_root,
        patch_size=args.patch_size,
        normalize=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = FavelaMulticlassSegModel(
        in_channels=12,
        num_classes=NUM_CLASSES,
        decoder_channels=args.decoder_channels,
        ppm_channels=args.ppm_channels,
        dropout=args.dropout,
    )

    weight_info = load_reben_weights_if_available(model, args.reben_repo_id)

    freeze_info = apply_freeze_mode(model, args.freeze_backbone_mode)

    model = model.to(device)

    class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    print("\nClass weights:")
    for i, w in enumerate(args.class_weights):
        print(f"  {i} {CLASS_NAMES[i]:>10s}: {w}")

    loss_fn = WeightedCEDiceLoss(
        class_weights=args.class_weights,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        include_background_in_dice=False,
    ).to(device)

    optimizer = make_optimizer(
        model=model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    amp = (not args.no_amp) and device.type == "cuda"
    scaler = GradScaler(enabled=amp)

    start_epoch = 1
    best_metric = -1.0

    if args.resume != "none":
        ckpt_name = "latest.pt" if args.resume == "latest" else "best.pt"
        start_epoch, best_metric = load_checkpoint_if_requested(
            run_dir / "checkpoints" / ckpt_name,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    summary = {
        "run_dir": str(run_dir),
        "instance_root": str(instance_root),
        "split_dir": str(split_dir),
        "class_names": CLASS_NAMES,
        "class_weights": args.class_weights,
        "augmentation_multipliers": multipliers,
        "weight_info": weight_info,
        "freeze_info": freeze_info,
    }
    save_json(summary, run_dir / "summary_initial.json")

    history_rows = []

    for epoch in range(start_epoch, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 80)

        train_summary = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp=amp,
            accum_steps=max(1, args.accum_steps),
            max_train_batches=args.max_train_batches,
            grad_clip_norm=args.grad_clip_norm,
        )

        val_summary, val_class_df, val_density_df = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            num_classes=NUM_CLASSES,
        )

        print("\nEpoch summary:")
        print(f"  train_loss: {train_summary['loss']:.4f}")
        print(f"  val_loss:   {val_summary['loss']:.4f}")
        print(f"  val_binary_iou_favela:        {val_summary['binary_iou_favela']:.4f}")
        print(f"  val_binary_dice_favela:       {val_summary['binary_dice_favela']:.4f}")
        print(f"  val_binary_precision_favela:  {val_summary['binary_precision_favela']:.4f}")
        print(f"  val_binary_recall_favela:     {val_summary['binary_recall_favela']:.4f}")
        print(f"  val_mean_favela_class_iou:    {val_summary['mean_favela_class_iou']:.4f}")

        row = {"epoch": epoch}
        for k, v in train_summary.items():
            row[f"train_{k}"] = v
        for k, v in val_summary.items():
            row[f"val_{k}"] = v
        history_rows.append(row)

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(run_dir / "history.csv", index=False)
        plot_history(history_df, run_dir / "figures")

        val_class_df.to_csv(run_dir / "metrics" / f"class_metrics_val_epoch{epoch:03d}.csv", index=False)
        val_density_df.to_csv(run_dir / "metrics" / f"density_metrics_val_epoch{epoch:03d}.csv", index=False)

        selection_metric = val_summary["binary_iou_favela"]

        save_checkpoint(
            run_dir / "checkpoints" / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_metric,
            args=args,
            extra={
                "val_summary": val_summary,
                "selection_metric": selection_metric,
                "weight_info": weight_info,
                "freeze_info": freeze_info,
            },
        )

        if selection_metric > best_metric:
            best_metric = selection_metric
            print(f"  New best checkpoint: val_binary_iou_favela={best_metric:.4f}")

            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                args=args,
                extra={
                    "val_summary": val_summary,
                    "selection_metric": selection_metric,
                    "weight_info": weight_info,
                    "freeze_info": freeze_info,
                },
            )

            val_class_df.to_csv(run_dir / "metrics" / "class_metrics_val_best.csv", index=False)
            val_density_df.to_csv(run_dir / "metrics" / "density_metrics_val_best.csv", index=False)

    print("\nLoading best checkpoint for final validation/test evaluation...")
    best_ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"], strict=True)

    val_summary, val_class_df, val_density_df = evaluate(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        num_classes=NUM_CLASSES,
    )

    test_summary, test_class_df, test_density_df = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        num_classes=NUM_CLASSES,
    )

    val_class_df.to_csv(run_dir / "metrics" / "class_metrics_val_final_best.csv", index=False)
    val_density_df.to_csv(run_dir / "metrics" / "density_metrics_val_final_best.csv", index=False)

    test_class_df.to_csv(run_dir / "metrics" / "class_metrics_test_final_best.csv", index=False)
    test_density_df.to_csv(run_dir / "metrics" / "density_metrics_test_final_best.csv", index=False)

    final_summary = {
        "run_dir": str(run_dir),
        "best_epoch": int(best_ckpt["epoch"]),
        "best_metric_val_binary_iou": float(best_ckpt["best_metric"]),
        "val": val_summary,
        "test": test_summary,
        "class_names": CLASS_NAMES,
        "class_weights": args.class_weights,
        "augmentation_multipliers": multipliers,
        "weight_info": weight_info,
        "freeze_info": freeze_info,
    }

    save_json(final_summary, run_dir / "summary_final.json")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Best epoch: {final_summary['best_epoch']}")

    print("\nValidation:")
    for k, v in val_summary.items():
        print(f"  {k}: {v:.6f}")

    print("\nTest:")
    for k, v in test_summary.items():
        print(f"  {k}: {v:.6f}")

    print("\nClass metrics, test:")
    print(test_class_df[["class_id", "class_name", "iou", "dice", "precision", "recall", "support_pixels"]].to_string(index=False))

    print("\nDensity metrics, test:")
    if not test_density_df.empty:
        print(test_density_df[[
            "density_class",
            "density_label",
            "n_patches",
            "binary_iou",
            "binary_dice",
            "binary_precision",
            "binary_recall",
            "binary_gt_pos_pct",
            "binary_pred_pos_pct",
            "binary_pred_minus_gt_pct",
        ]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()