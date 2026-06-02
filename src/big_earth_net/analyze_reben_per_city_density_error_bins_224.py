#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_reben_per_city_density_error_bins_224.py

Main objective
--------------
Analyze per-patch prediction metrics from the reBEN ResNet18 + UPerNet
visualization script.

This script does not train or run inference. It reads:

    per_patch_metrics_<split>.csv

created by:

    src/big_earth_net/visualize_reben_predictions_224.py

and produces summary tables and diagnostic plots by:

    - city
    - region
    - label-density bin
    - prediction-density bin
    - error mode
    - city x label-density bin

Main outputs
------------
analysis_per_city_density_error_bins_<split>/
    overall_summary.csv
    by_region_metrics.csv
    by_city_metrics.csv
    by_label_density_bin_metrics.csv
    by_prediction_density_bin_metrics.csv
    by_city_label_density_bin_metrics.csv
    error_mode_counts.csv
    error_mode_counts_by_city.csv
    hard_false_positive_patches.csv
    hard_false_negative_patches.csv
    severe_overprediction_patches.csv
    severe_underprediction_patches.csv
    tiny_fragment_missed_patches.csv
    good_positive_patches.csv
    worst_100_positive_patches.csv
    best_100_positive_patches.csv
    diagnostic_report.md
    figures/*.png

Example
-------
python src\\big_earth_net\\analyze_reben_per_city_density_error_bins_224.py `
  --metrics-csv "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired/experiments/big_earth_net/reben_resnet18_upernet_s1s2_train_region_covered_ps224/reben_resnet18_upernet_s1s2_train_region_covered_epochs30_bs2_acc4_fullfinetune_posw10/visual_predictions/test_best_thr0p450/per_patch_metrics_test.csv" `
  --overwrite

Alternative auto-discovery
--------------------------
python src\\big_earth_net\\analyze_reben_per_city_density_error_bins_224.py `
  --run-dir "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired/experiments/big_earth_net/reben_resnet18_upernet_s1s2_train_region_covered_ps224/reben_resnet18_upernet_s1s2_train_region_covered_epochs30_bs2_acc4_fullfinetune_posw10" `
  --split test `
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
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
        "[ERROR] matplotlib is required.\n"
        "Install it with:\n"
        "    pip install matplotlib\n\n"
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


def path_to_str(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return str(path).replace("\\", "/")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        fail(
            "Output directory already exists and is not empty:\n"
            f"{path_to_str(path)}\n\n"
            "Use --overwrite to update it."
        )
    path.mkdir(parents=True, exist_ok=True)


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


# ---------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------

def infer_split_from_metrics_path(metrics_csv: Path) -> str:
    name = metrics_csv.name.lower()

    if "train" in name:
        return "train"
    if "val" in name:
        return "val"
    if "test" in name:
        return "test"

    return "unknown"


def find_latest_metrics_csv(run_dir: Path, split: str) -> Path:
    visual_root = run_dir / "visual_predictions"

    if not visual_root.exists():
        fail(
            "Could not find visual_predictions directory:\n"
            f"{path_to_str(visual_root)}\n\n"
            "Either pass --metrics-csv explicitly or run visualize_reben_predictions_224.py first."
        )

    pattern = f"**/per_patch_metrics_{split}.csv"
    candidates = sorted(visual_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        fail(
            f"No per_patch_metrics_{split}.csv found under:\n"
            f"{path_to_str(visual_root)}"
        )

    return candidates[0]


def resolve_metrics_csv(args: argparse.Namespace) -> Path:
    if args.metrics_csv:
        path = Path(args.metrics_csv)
        if not path.exists():
            fail(f"--metrics-csv does not exist:\n{path_to_str(path)}")
        return path

    if not args.run_dir:
        fail("You must provide either --metrics-csv or --run-dir.")

    return find_latest_metrics_csv(Path(args.run_dir), args.split)


def default_output_dir(metrics_csv: Path, split: str) -> Path:
    return metrics_csv.parent / f"analysis_per_city_density_error_bins_{split}"


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "patch_id",
    "city",
    "region",
    "tp",
    "fp",
    "fn",
    "tn",
    "iou_favela",
    "dice_favela",
    "precision_favela",
    "recall_favela",
    "iou_no_favela",
    "precision_no_favela",
    "recall_no_favela",
    "macro_iou",
    "pred_pos_pct",
    "gt_pos_pct",
    "fp_pct",
    "fn_pct",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        fail(
            "Input metrics CSV is missing required columns:\n"
            f"{missing}\n\n"
            "Available columns:\n"
            f"{list(df.columns)}\n\n"
            "Make sure the file was created by the updated visualize_reben_predictions_224.py script."
        )


def safe_div(num: float, den: float, eps: float = 1e-8) -> float:
    return float(num) / float(den + eps)


def compute_global_metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> Dict[str, float]:
    eps = 1e-8

    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    tn = float(tn)

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
    macro_precision = 0.5 * (precision_favela + precision_no_favela)
    macro_recall = 0.5 * (recall_favela + recall_no_favela)

    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    n_pixels = tp + fp + fn + tn
    pred_pos = tp + fp
    gt_pos = tp + fn

    return {
        "global_iou_favela": float(iou_favela),
        "global_dice_favela": float(dice_favela),
        "global_precision_favela": float(precision_favela),
        "global_recall_favela": float(recall_favela),
        "global_iou_no_favela": float(iou_no_favela),
        "global_dice_no_favela": float(dice_no_favela),
        "global_precision_no_favela": float(precision_no_favela),
        "global_recall_no_favela": float(recall_no_favela),
        "global_macro_iou": float(macro_iou),
        "global_macro_dice": float(macro_dice),
        "global_macro_precision": float(macro_precision),
        "global_macro_recall": float(macro_recall),
        "global_accuracy": float(accuracy),
        "global_pred_pos_pct": float(100.0 * pred_pos / (n_pixels + eps)),
        "global_gt_pos_pct": float(100.0 * gt_pos / (n_pixels + eps)),
        "global_pred_minus_gt_pct": float(100.0 * (pred_pos - gt_pos) / (n_pixels + eps)),
    }


def numeric_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if col in {"patch_id", "city", "region", "category", "label_density_bin", "prediction_density_bin", "dominant_error_mode"}:
            continue

        try:
            out[col] = pd.to_numeric(out[col], errors="ignore")
        except Exception:
            pass

    if "pred_minus_gt_pct" not in out.columns:
        out["pred_minus_gt_pct"] = out["pred_pos_pct"] - out["gt_pos_pct"]

    if "gt_minus_pred_pct" not in out.columns:
        out["gt_minus_pred_pct"] = out["gt_pos_pct"] - out["pred_pos_pct"]

    return out


# ---------------------------------------------------------------------
# Binning and error modes
# ---------------------------------------------------------------------

def label_density_bin(gt_pos_pct: float) -> str:
    x = float(gt_pos_pct)

    if x <= 0.0:
        return "00_empty_gt_0"
    if x <= 0.5:
        return "01_tiny_0_0p5"
    if x <= 1.0:
        return "02_tiny_0p5_1"
    if x <= 5.0:
        return "03_low_1_5"
    if x <= 10.0:
        return "04_moderate_5_10"
    if x <= 20.0:
        return "05_medium_10_20"
    if x <= 40.0:
        return "06_high_20_40"
    if x <= 60.0:
        return "07_very_high_40_60"
    return "08_extreme_gt_60"


def prediction_density_bin(pred_pos_pct: float) -> str:
    x = float(pred_pos_pct)

    if x <= 0.0:
        return "00_empty_pred_0"
    if x <= 0.5:
        return "01_tiny_0_0p5"
    if x <= 1.0:
        return "02_tiny_0p5_1"
    if x <= 5.0:
        return "03_low_1_5"
    if x <= 10.0:
        return "04_moderate_5_10"
    if x <= 20.0:
        return "05_medium_10_20"
    if x <= 40.0:
        return "06_high_20_40"
    if x <= 60.0:
        return "07_very_high_40_60"
    return "08_extreme_gt_60"


def add_bins_and_error_flags(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()

    out["label_density_bin"] = out["gt_pos_pct"].apply(label_density_bin)
    out["prediction_density_bin"] = out["pred_pos_pct"].apply(prediction_density_bin)

    out["is_empty_gt"] = out["gt_pos_pct"] <= float(args.empty_gt_threshold_pct)
    out["is_tiny_positive"] = (out["gt_pos_pct"] > float(args.empty_gt_threshold_pct)) & (
        out["gt_pos_pct"] <= float(args.tiny_gt_threshold_pct)
    )
    out["is_positive_patch"] = out["gt_pos_pct"] > float(args.positive_gt_threshold_pct)

    out["is_good_positive"] = (
        (out["gt_pos_pct"] >= float(args.good_min_gt_pct))
        & (out["iou_favela"] >= float(args.good_iou_threshold))
    )

    out["is_hard_false_positive"] = (
        (out["gt_pos_pct"] <= float(args.hard_fp_max_gt_pct))
        & (out["pred_pos_pct"] >= float(args.hard_fp_min_pred_pct))
    )

    out["is_moderate_false_positive"] = (
        (out["gt_pos_pct"] <= float(args.moderate_fp_max_gt_pct))
        & (out["pred_pos_pct"] >= float(args.moderate_fp_min_pred_pct))
    )

    out["is_severe_overprediction"] = (
        out["pred_minus_gt_pct"] >= float(args.severe_overprediction_margin_pct)
    )

    out["is_hard_false_negative"] = (
        (out["gt_pos_pct"] >= float(args.hard_fn_min_gt_pct))
        & (out["recall_favela"] <= float(args.hard_fn_max_recall))
    )

    out["is_severe_underprediction"] = (
        out["gt_minus_pred_pct"] >= float(args.severe_underprediction_margin_pct)
    )

    out["is_tiny_fragment_missed"] = (
        (out["gt_pos_pct"] > float(args.empty_gt_threshold_pct))
        & (out["gt_pos_pct"] <= float(args.tiny_gt_threshold_pct))
        & (out["recall_favela"] <= float(args.tiny_missed_max_recall))
    )

    out["is_boundary_or_fragment_case"] = (
        (out["gt_pos_pct"] > float(args.empty_gt_threshold_pct))
        & (out["gt_pos_pct"] <= float(args.fragment_gt_threshold_pct))
    )

    def dominant_mode(row: pd.Series) -> str:
        if bool(row["is_good_positive"]):
            return "good_positive"

        if bool(row["is_hard_false_positive"]):
            return "hard_false_positive"

        if bool(row["is_hard_false_negative"]):
            return "hard_false_negative"

        if bool(row["is_severe_overprediction"]):
            return "severe_overprediction"

        if bool(row["is_severe_underprediction"]):
            return "severe_underprediction"

        if bool(row["is_tiny_fragment_missed"]):
            return "tiny_fragment_missed"

        if bool(row["is_empty_gt"]) and row["pred_pos_pct"] <= 1.0:
            return "good_empty_or_near_empty"

        if bool(row["is_tiny_positive"]):
            return "tiny_positive_other"

        if row["iou_favela"] < 0.1 and row["gt_pos_pct"] >= 1.0:
            return "low_iou_positive_other"

        return "mixed_or_average"

    out["dominant_error_mode"] = out.apply(dominant_mode, axis=1)

    return out


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def aggregate_group(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    grouped = df.groupby(list(group_cols), dropna=False)

    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row: Dict[str, Any] = {}
        for col, key in zip(group_cols, keys):
            row[col] = key

        tp = float(g["tp"].sum())
        fp = float(g["fp"].sum())
        fn = float(g["fn"].sum())
        tn = float(g["tn"].sum())

        row.update(
            {
                "n_patches": int(len(g)),
                "n_cities": int(g["city"].nunique()) if "city" in g.columns else None,
                "n_regions": int(g["region"].nunique()) if "region" in g.columns else None,

                "patch_mean_iou_favela": float(g["iou_favela"].mean()),
                "patch_median_iou_favela": float(g["iou_favela"].median()),
                "patch_p25_iou_favela": float(g["iou_favela"].quantile(0.25)),
                "patch_p75_iou_favela": float(g["iou_favela"].quantile(0.75)),

                "patch_mean_dice_favela": float(g["dice_favela"].mean()),
                "patch_mean_precision_favela": float(g["precision_favela"].mean()),
                "patch_mean_recall_favela": float(g["recall_favela"].mean()),

                "patch_mean_iou_no_favela": float(g["iou_no_favela"].mean()),
                "patch_mean_macro_iou": float(g["macro_iou"].mean()),

                "mean_gt_pos_pct": float(g["gt_pos_pct"].mean()),
                "median_gt_pos_pct": float(g["gt_pos_pct"].median()),
                "mean_pred_pos_pct": float(g["pred_pos_pct"].mean()),
                "median_pred_pos_pct": float(g["pred_pos_pct"].median()),
                "mean_pred_minus_gt_pct": float(g["pred_minus_gt_pct"].mean()),

                "mean_fp_pct": float(g["fp_pct"].mean()),
                "mean_fn_pct": float(g["fn_pct"].mean()),

                "fraction_good_positive": float(g["is_good_positive"].mean()),
                "fraction_hard_false_positive": float(g["is_hard_false_positive"].mean()),
                "fraction_hard_false_negative": float(g["is_hard_false_negative"].mean()),
                "fraction_tiny_fragment_missed": float(g["is_tiny_fragment_missed"].mean()),

                "total_tp": tp,
                "total_fp": fp,
                "total_fn": fn,
                "total_tn": tn,
            }
        )

        row.update(compute_global_metrics_from_counts(tp, fp, fn, tn))

        rows.append(row)

    out = pd.DataFrame(rows)

    if "global_iou_favela" in out.columns:
        out = out.sort_values("global_iou_favela", ascending=True).reset_index(drop=True)

    return out


def error_mode_counts(df: pd.DataFrame, group_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if group_cols is None or len(group_cols) == 0:
        counts = (
            df["dominant_error_mode"]
            .value_counts()
            .rename_axis("dominant_error_mode")
            .reset_index(name="n_patches")
        )
        counts["fraction"] = counts["n_patches"] / max(1, len(df))
        return counts

    counts = (
        df.groupby(list(group_cols) + ["dominant_error_mode"], dropna=False)
        .size()
        .reset_index(name="n_patches")
    )

    totals = (
        df.groupby(list(group_cols), dropna=False)
        .size()
        .reset_index(name="total_patches")
    )

    out = counts.merge(totals, on=list(group_cols), how="left")
    out["fraction"] = out["n_patches"] / out["total_patches"].clip(lower=1)

    return out.sort_values(list(group_cols) + ["n_patches"], ascending=[True] * len(group_cols) + [False])


# ---------------------------------------------------------------------
# Patch selection outputs
# ---------------------------------------------------------------------

PATCH_EXPORT_COLUMNS = [
    "patch_id",
    "city",
    "region",
    "label_density_bin",
    "prediction_density_bin",
    "dominant_error_mode",
    "iou_favela",
    "dice_favela",
    "precision_favela",
    "recall_favela",
    "iou_no_favela",
    "macro_iou",
    "pred_pos_pct",
    "gt_pos_pct",
    "pred_minus_gt_pct",
    "gt_minus_pred_pct",
    "fp_pct",
    "fn_pct",
    "tp",
    "fp",
    "fn",
    "tn",
]


def export_patch_lists(df: pd.DataFrame, output_dir: Path, top_n: int) -> Dict[str, int]:
    ensure_dir(output_dir)

    exported: Dict[str, int] = {}

    def export(name: str, frame: pd.DataFrame, sort_cols: Sequence[str], ascending: Sequence[bool]) -> None:
        cols = [c for c in PATCH_EXPORT_COLUMNS if c in frame.columns]
        out = frame.sort_values(list(sort_cols), ascending=list(ascending)).head(int(top_n)).copy()
        path = output_dir / f"{name}.csv"
        out[cols].to_csv(path, index=False)
        exported[name] = int(len(out))

    positive = df[df["gt_pos_pct"] > 0].copy()

    export(
        "hard_false_positive_patches",
        df[df["is_hard_false_positive"]].copy(),
        ["pred_pos_pct", "gt_pos_pct"],
        [False, True],
    )

    export(
        "hard_false_negative_patches",
        df[df["is_hard_false_negative"]].copy(),
        ["gt_pos_pct", "recall_favela"],
        [False, True],
    )

    export(
        "severe_overprediction_patches",
        df[df["is_severe_overprediction"]].copy(),
        ["pred_minus_gt_pct"],
        [False],
    )

    export(
        "severe_underprediction_patches",
        df[df["is_severe_underprediction"]].copy(),
        ["gt_minus_pred_pct"],
        [False],
    )

    export(
        "tiny_fragment_missed_patches",
        df[df["is_tiny_fragment_missed"]].copy(),
        ["gt_pos_pct"],
        [False],
    )

    export(
        "good_positive_patches",
        df[df["is_good_positive"]].copy(),
        ["iou_favela", "gt_pos_pct"],
        [False, False],
    )

    export(
        "worst_100_positive_patches",
        positive,
        ["iou_favela", "gt_pos_pct"],
        [True, False],
    )

    export(
        "best_100_positive_patches",
        positive,
        ["iou_favela", "gt_pos_pct"],
        [False, False],
    )

    export(
        "largest_label_patches",
        df.copy(),
        ["gt_pos_pct"],
        [False],
    )

    export(
        "largest_prediction_patches",
        df.copy(),
        ["pred_pos_pct"],
        [False],
    )

    return exported


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def save_bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    top_n: Optional[int] = None,
    rotate: int = 45,
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        warn(f"Skipping plot {output_path.name}: missing {x_col} or {y_col}")
        return

    plot_df = df.copy()

    if top_n is not None and len(plot_df) > top_n:
        plot_df = plot_df.head(top_n)

    ensure_dir(output_path.parent)

    plt.figure(figsize=(max(9, 0.45 * len(plot_df)), 5))
    plt.bar(plot_df[x_col].astype(str), plot_df[y_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(x_col)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_grouped_positive_pct_plot(df: pd.DataFrame, x_col: str, output_path: Path, title: str) -> None:
    required = [x_col, "mean_gt_pos_pct", "mean_pred_pos_pct"]

    if df.empty or any(c not in df.columns for c in required):
        warn(f"Skipping plot {output_path.name}: missing required columns")
        return

    plot_df = df.copy()
    ensure_dir(output_path.parent)

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(max(9, 0.45 * len(plot_df)), 5))
    plt.bar(x - width / 2, plot_df["mean_gt_pos_pct"], width, label="GT favela %")
    plt.bar(x + width / 2, plot_df["mean_pred_pos_pct"], width, label="Predicted favela %")
    plt.title(title)
    plt.ylabel("Favela pixels (%)")
    plt.xlabel(x_col)
    plt.xticks(x, plot_df[x_col].astype(str), rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_error_mode_stacked_plot(error_city_df: pd.DataFrame, output_path: Path) -> None:
    if error_city_df.empty:
        warn(f"Skipping plot {output_path.name}: empty error city dataframe")
        return

    pivot = error_city_df.pivot_table(
        index="city",
        columns="dominant_error_mode",
        values="fraction",
        aggfunc="sum",
        fill_value=0.0,
    )

    if pivot.empty:
        return

    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    ensure_dir(output_path.parent)

    ax = pivot.plot(kind="bar", stacked=True, figsize=(max(10, 0.5 * len(pivot)), 6))
    ax.set_title("Dominant error mode fractions by city")
    ax.set_ylabel("Fraction of patches")
    ax.set_xlabel("city")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_scatter_pred_vs_gt(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return

    ensure_dir(output_path.parent)

    plt.figure(figsize=(6, 6))
    plt.scatter(df["gt_pos_pct"], df["pred_pos_pct"], s=10, alpha=0.45)
    lim = max(float(df["gt_pos_pct"].max()), float(df["pred_pos_pct"].max()), 1.0)
    plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    plt.xlabel("Ground-truth favela pixels (%)")
    plt.ylabel("Predicted favela pixels (%)")
    plt.title("Patch-level predicted vs ground-truth favela percentage")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_figures(
    output_dir: Path,
    city_df: pd.DataFrame,
    region_df: pd.DataFrame,
    density_df: pd.DataFrame,
    pred_density_df: pd.DataFrame,
    error_city_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> None:
    figures_dir = output_dir / "figures"
    ensure_dir(figures_dir)

    city_sorted = city_df.sort_values("global_iou_favela", ascending=True).copy()
    city_sorted["city_region"] = city_sorted["city"].astype(str) + " [" + city_sorted["region"].astype(str) + "]"

    save_bar_plot(
        city_sorted,
        x_col="city_region",
        y_col="global_iou_favela",
        title="Global favela IoU by city",
        ylabel="Global favela IoU",
        output_path=figures_dir / "city_global_iou_favela.png",
    )

    save_bar_plot(
        city_sorted,
        x_col="city_region",
        y_col="patch_mean_iou_favela",
        title="Mean patch favela IoU by city",
        ylabel="Mean patch favela IoU",
        output_path=figures_dir / "city_patch_mean_iou_favela.png",
    )

    save_grouped_positive_pct_plot(
        city_sorted,
        x_col="city_region",
        output_path=figures_dir / "city_gt_vs_pred_favela_pct.png",
        title="Mean predicted vs ground-truth favela percentage by city",
    )

    save_bar_plot(
        region_df.sort_values("global_iou_favela", ascending=True),
        x_col="region",
        y_col="global_iou_favela",
        title="Global favela IoU by region",
        ylabel="Global favela IoU",
        output_path=figures_dir / "region_global_iou_favela.png",
    )

    save_bar_plot(
        density_df.sort_values("label_density_bin"),
        x_col="label_density_bin",
        y_col="global_iou_favela",
        title="Global favela IoU by label-density bin",
        ylabel="Global favela IoU",
        output_path=figures_dir / "label_density_global_iou_favela.png",
    )

    save_bar_plot(
        density_df.sort_values("label_density_bin"),
        x_col="label_density_bin",
        y_col="n_patches",
        title="Number of patches by label-density bin",
        ylabel="Number of patches",
        output_path=figures_dir / "label_density_patch_count.png",
    )

    save_bar_plot(
        pred_density_df.sort_values("prediction_density_bin"),
        x_col="prediction_density_bin",
        y_col="n_patches",
        title="Number of patches by prediction-density bin",
        ylabel="Number of patches",
        output_path=figures_dir / "prediction_density_patch_count.png",
    )

    save_error_mode_stacked_plot(
        error_city_df=error_city_df,
        output_path=figures_dir / "error_mode_fraction_by_city.png",
    )

    save_scatter_pred_vs_gt(
        df=full_df,
        output_path=figures_dir / "patch_pred_vs_gt_favela_pct.png",
    )


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


def compact_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    return df[present].copy()


def write_report(
    output_dir: Path,
    metrics_csv: Path,
    split: str,
    full_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    region_df: pd.DataFrame,
    city_df: pd.DataFrame,
    density_df: pd.DataFrame,
    error_counts_df: pd.DataFrame,
    error_city_df: pd.DataFrame,
    exported_counts: Dict[str, int],
) -> None:
    report_path = output_dir / "diagnostic_report.md"

    lines: List[str] = []

    lines.append(f"# reBEN Per-City / Density / Error-Bin Diagnostic Report ({split})")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "This report summarizes per-patch segmentation behaviour in order to diagnose why "
        "the global favela IoU is limited despite strong performance on selected patches."
    )
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- Metrics CSV: `{path_to_str(metrics_csv)}`")
    lines.append(f"- Number of patches: `{len(full_df)}`")
    lines.append(f"- Output directory: `{path_to_str(output_dir)}`")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(markdown_table(overall_df))
    lines.append("")
    lines.append("## Region Summary")
    lines.append("")

    region_cols = [
        "region",
        "n_patches",
        "global_iou_favela",
        "global_dice_favela",
        "global_precision_favela",
        "global_recall_favela",
        "global_iou_no_favela",
        "global_macro_iou",
        "global_gt_pos_pct",
        "global_pred_pos_pct",
        "global_pred_minus_gt_pct",
        "fraction_hard_false_positive",
        "fraction_hard_false_negative",
    ]
    lines.append(markdown_table(compact_cols(region_df, region_cols)))
    lines.append("")
    lines.append("## Worst Cities by Global Favela IoU")
    lines.append("")

    city_cols = [
        "region",
        "city",
        "n_patches",
        "global_iou_favela",
        "global_dice_favela",
        "global_precision_favela",
        "global_recall_favela",
        "global_iou_no_favela",
        "global_macro_iou",
        "global_gt_pos_pct",
        "global_pred_pos_pct",
        "global_pred_minus_gt_pct",
        "fraction_hard_false_positive",
        "fraction_hard_false_negative",
        "fraction_tiny_fragment_missed",
    ]
    lines.append(markdown_table(compact_cols(city_df, city_cols), max_rows=30))
    lines.append("")
    lines.append("## Label-Density Bin Summary")
    lines.append("")

    density_cols = [
        "label_density_bin",
        "n_patches",
        "global_iou_favela",
        "global_dice_favela",
        "global_precision_favela",
        "global_recall_favela",
        "global_gt_pos_pct",
        "global_pred_pos_pct",
        "fraction_hard_false_positive",
        "fraction_hard_false_negative",
        "fraction_tiny_fragment_missed",
    ]
    lines.append(markdown_table(compact_cols(density_df.sort_values("label_density_bin"), density_cols)))
    lines.append("")
    lines.append("## Error Mode Counts")
    lines.append("")
    lines.append(markdown_table(error_counts_df))
    lines.append("")
    lines.append("## Most Frequent Error Modes by City")
    lines.append("")

    error_city_compact = error_city_df.sort_values(["city", "fraction"], ascending=[True, False])
    lines.append(markdown_table(error_city_compact, max_rows=100))
    lines.append("")
    lines.append("## Exported Patch Lists")
    lines.append("")
    export_rows = [{"patch_list": k, "n_rows": v} for k, v in exported_counts.items()]
    lines.append(markdown_table(pd.DataFrame(export_rows)))
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("### Hard false positives")
    lines.append("")
    lines.append(
        "Patches where ground-truth favela percentage is very small, but the model predicts a large favela area. "
        "These usually indicate confusion between favela and dense formal built-up fabric."
    )
    lines.append("")
    lines.append("### Hard false negatives")
    lines.append("")
    lines.append(
        "Patches where the ground-truth favela area is large, but the model recall is very low. "
        "These can indicate city-specific domain shift, overly broad labels, or favela types not learned by the model."
    )
    lines.append("")
    lines.append("### Tiny fragment misses")
    lines.append("")
    lines.append(
        "Patches with very small ground-truth favela fragments. IoU is unstable here: missing a few pixels can produce zero IoU."
    )
    lines.append("")
    lines.append("### Recommended next actions")
    lines.append("")
    lines.append("1. Audit cities with high hard-false-negative rates, especially if labels are very large and visually heterogeneous.")
    lines.append("2. Use hard false positives as a hard-negative mining pool for retraining.")
    lines.append("3. Report metrics stratified by label-density bin, not only global IoU.")
    lines.append("4. Consider separate evaluation excluding tiny fragments to understand performance on meaningful positive patches.")
    lines.append("5. If dense urban false positives dominate, test lower positive weight or a loss with stronger false-positive penalty.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    metrics_csv = resolve_metrics_csv(args)
    split = args.split if args.split else infer_split_from_metrics_path(metrics_csv)

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(metrics_csv, split)
    ensure_output_dir(output_dir, overwrite=bool(args.overwrite))

    banner("Analyze reBEN per-city / density / error-bin metrics")

    log("INFO", f"Metrics CSV: {path_to_str(metrics_csv)}")
    log("INFO", f"Split:       {split}")
    log("INFO", f"Output dir:  {path_to_str(output_dir)}")

    df = pd.read_csv(metrics_csv)
    validate_columns(df)
    df = numeric_clean(df)
    df = add_bins_and_error_flags(df, args)

    enriched_path = output_dir / "per_patch_metrics_with_bins_and_error_modes.csv"
    df.to_csv(enriched_path, index=False)
    log("OK", f"Saved enriched per-patch metrics:\n{path_to_str(enriched_path)}")

    # Overall summary from global counts.
    total_tp = float(df["tp"].sum())
    total_fp = float(df["fp"].sum())
    total_fn = float(df["fn"].sum())
    total_tn = float(df["tn"].sum())

    overall = {
        "split": split,
        "n_patches": int(len(df)),
        "n_cities": int(df["city"].nunique()),
        "n_regions": int(df["region"].nunique()),
        "patch_mean_iou_favela": float(df["iou_favela"].mean()),
        "patch_median_iou_favela": float(df["iou_favela"].median()),
        "patch_mean_dice_favela": float(df["dice_favela"].mean()),
        "patch_mean_precision_favela": float(df["precision_favela"].mean()),
        "patch_mean_recall_favela": float(df["recall_favela"].mean()),
        "patch_mean_gt_pos_pct": float(df["gt_pos_pct"].mean()),
        "patch_mean_pred_pos_pct": float(df["pred_pos_pct"].mean()),
        "fraction_good_positive": float(df["is_good_positive"].mean()),
        "fraction_hard_false_positive": float(df["is_hard_false_positive"].mean()),
        "fraction_hard_false_negative": float(df["is_hard_false_negative"].mean()),
        "fraction_tiny_fragment_missed": float(df["is_tiny_fragment_missed"].mean()),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
    }
    overall.update(compute_global_metrics_from_counts(total_tp, total_fp, total_fn, total_tn))
    overall_df = pd.DataFrame([overall])

    region_df = aggregate_group(df, ["region"])
    city_df = aggregate_group(df, ["region", "city"])
    density_df = aggregate_group(df, ["label_density_bin"])
    pred_density_df = aggregate_group(df, ["prediction_density_bin"])
    city_density_df = aggregate_group(df, ["region", "city", "label_density_bin"])

    error_counts_df = error_mode_counts(df)
    error_city_df = error_mode_counts(df, ["region", "city"])

    overall_df.to_csv(output_dir / "overall_summary.csv", index=False)
    region_df.to_csv(output_dir / "by_region_metrics.csv", index=False)
    city_df.to_csv(output_dir / "by_city_metrics.csv", index=False)
    density_df.to_csv(output_dir / "by_label_density_bin_metrics.csv", index=False)
    pred_density_df.to_csv(output_dir / "by_prediction_density_bin_metrics.csv", index=False)
    city_density_df.to_csv(output_dir / "by_city_label_density_bin_metrics.csv", index=False)
    error_counts_df.to_csv(output_dir / "error_mode_counts.csv", index=False)
    error_city_df.to_csv(output_dir / "error_mode_counts_by_city.csv", index=False)

    exported_counts = export_patch_lists(df, output_dir, top_n=int(args.top_n_patches))

    save_figures(
        output_dir=output_dir,
        city_df=city_df,
        region_df=region_df,
        density_df=density_df,
        pred_density_df=pred_density_df,
        error_city_df=error_city_df,
        full_df=df,
    )

    write_report(
        output_dir=output_dir,
        metrics_csv=metrics_csv,
        split=split,
        full_df=df,
        overall_df=overall_df,
        region_df=region_df,
        city_df=city_df,
        density_df=density_df,
        error_counts_df=error_counts_df,
        error_city_df=error_city_df,
        exported_counts=exported_counts,
    )

    write_json(
        output_dir / "analysis_config.json",
        {
            "metrics_csv": path_to_str(metrics_csv),
            "split": split,
            "output_dir": path_to_str(output_dir),
            "args": vars(args),
            "n_patches": int(len(df)),
            "n_cities": int(df["city"].nunique()),
            "n_regions": int(df["region"].nunique()),
            "overall": overall,
            "exported_counts": exported_counts,
        },
    )

    banner("Completed")
    log("OK", f"Output directory:\n{path_to_str(output_dir)}")
    log("OK", f"Report:\n{path_to_str(output_dir / 'diagnostic_report.md')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze reBEN per-patch metrics by city, density bins, and error modes."
    )

    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Explicit path to per_patch_metrics_<split>.csv from visualize_reben_predictions_224.py.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Training run directory. Used to auto-discover latest visual_predictions/*/per_patch_metrics_<split>.csv.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "unknown"],
        default="test",
        help="Split name for auto-discovery and report naming.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: metrics CSV parent / analysis_per_city_density_error_bins_<split>.",
    )

    parser.add_argument("--top-n-patches", type=int, default=100)

    # Error-mode thresholds.
    parser.add_argument("--empty-gt-threshold-pct", type=float, default=0.0)
    parser.add_argument("--positive-gt-threshold-pct", type=float, default=0.0)
    parser.add_argument("--tiny-gt-threshold-pct", type=float, default=1.0)
    parser.add_argument("--fragment-gt-threshold-pct", type=float, default=2.0)

    parser.add_argument("--good-min-gt-pct", type=float, default=1.0)
    parser.add_argument("--good-iou-threshold", type=float, default=0.50)

    parser.add_argument("--hard-fp-max-gt-pct", type=float, default=1.0)
    parser.add_argument("--hard-fp-min-pred-pct", type=float, default=20.0)

    parser.add_argument("--moderate-fp-max-gt-pct", type=float, default=5.0)
    parser.add_argument("--moderate-fp-min-pred-pct", type=float, default=10.0)

    parser.add_argument("--severe-overprediction-margin-pct", type=float, default=20.0)

    parser.add_argument("--hard-fn-min-gt-pct", type=float, default=20.0)
    parser.add_argument("--hard-fn-max-recall", type=float, default=0.20)

    parser.add_argument("--severe-underprediction-margin-pct", type=float, default=20.0)

    parser.add_argument("--tiny-missed-max-recall", type=float, default=0.05)

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())