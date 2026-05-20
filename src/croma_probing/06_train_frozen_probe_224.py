#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
06_train_frozen_probe_224.py

Train frozen probes on CROMA embeddings for Instance C.

This script implements Criterion 1:

    Downstream frozen-probe performance

Task:
    Predict patch_label_binary from frozen CROMA embeddings.

Input embeddings:
    metadata/croma_probing/full_embeddings/
        croma_embeddings_s2_ps224_st112_cover.npz
        croma_embeddings_s1_snap_vv_vh_ps224_st112_cover.npz
        croma_embeddings_s1_rtc_vv_vh_ps224_st112_cover.npz
        croma_embeddings_s2_s1_snap_vv_vh_ps224_st112_cover.npz
        croma_embeddings_s2_s1_rtc_vv_vh_ps224_st112_cover.npz

Primary metric:
    Average Precision

Supporting metrics:
    ROC-AUC
    F1
    Precision
    Recall
    Balanced Accuracy
    Accuracy

Probe models:
    logistic_regression
    linear_svm
    small_mlp

Spatial fold modes:
    leave_one_region
    leave_one_city

Recommended first run, region-wise:

python src/croma_probing/06_train_frozen_probe_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --fold-modes leave_one_region `
  --probes logistic_regression linear_svm small_mlp `
  --overwrite

Recommended full robustness run:

python src/croma_probing/06_train_frozen_probe_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --fold-modes leave_one_region leave_one_city `
  --probes logistic_regression linear_svm small_mlp `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
except ImportError as exc:
    raise SystemExit(
        "[ERROR] scikit-learn is required.\n"
        "Install it with:\n"
        "    pip install scikit-learn\n\n"
        f"Original error: {exc}"
    )


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    log("ERROR", message)
    raise SystemExit(exit_code)


def path_to_str(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return str(path).replace("\\", "/")


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def slugify_for_filename(value: str) -> str:
    """
    Convert a string into a safe filename component.
    """
    text = str(value).strip().lower()
    text = text.replace(" ", "_")
    text = text.replace("-", "_")

    allowed = []
    for char in text:
        if char.isalnum() or char == "_":
            allowed.append(char)
        else:
            allowed.append("_")

    text = "".join(allowed)

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_")


def fold_modes_slug(fold_modes: Sequence[str]) -> str:
    """
    Build a stable filename slug from the requested fold modes.

    Examples:
        ["leave_one_region"] -> "leave_one_region"
        ["leave_one_city"] -> "leave_one_city"
        ["leave_one_region", "leave_one_city"] -> "leave_one_region__leave_one_city"
    """
    return "__".join(slugify_for_filename(x) for x in fold_modes)

def safe_float(value: object, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def round_float(value: float, digits: int = 8) -> float:
    value = safe_float(value, 0.0)
    return round(value, digits)


# ---------------------------------------------------------------------
# CSV / JSON / Markdown
# ---------------------------------------------------------------------

def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    overwrite: bool,
    fieldnames: Optional[List[str]] = None,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    if fieldnames is None:
        if not rows:
            fail(f"No rows to write: {path_to_str(path)}")
        fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(
    path: Path,
    summary: Dict[str, object],
    result_rows: List[Dict[str, object]],
    aggregate_rows: List[Dict[str, object]],
    comparison_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Frozen CROMA probe results")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Finished UTC: `{summary['finished_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Embedding dir: `{summary['embedding_dir']}`")
    lines.append(f"- Output dir: `{summary['output_dir']}`")
    lines.append(f"- Modalities: `{';'.join(summary['modalities'])}`")
    lines.append(f"- Probes: `{';'.join(summary['probes'])}`")
    lines.append(f"- Fold modes: `{';'.join(summary['fold_modes'])}`")
    lines.append(f"- Total result rows: `{summary['n_result_rows']}`")
    lines.append(f"- Failed rows: `{summary['n_failed_rows']}`")
    lines.append("")

    lines.append("## Best aggregate rows by Average Precision")
    lines.append("")
    lines.append("| fold mode | probe | modality | folds | AP mean | AP std | ROC-AUC mean | F1 mean | precision mean | recall mean | bal acc mean |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    sorted_agg = sorted(
        aggregate_rows,
        key=lambda r: (
            str(r["fold_mode"]),
            str(r["probe"]),
            -safe_float(r["average_precision_mean"]),
        ),
    )

    for row in sorted_agg:
        lines.append(
            f"| {row['fold_mode']} | "
            f"{row['probe']} | "
            f"{row['modality']} | "
            f"{row['n_folds']} | "
            f"{row['average_precision_mean']} | "
            f"{row['average_precision_std']} | "
            f"{row['roc_auc_mean']} | "
            f"{row['f1_mean']} | "
            f"{row['precision_mean']} | "
            f"{row['recall_mean']} | "
            f"{row['balanced_accuracy_mean']} |"
        )

    lines.append("")
    lines.append("## RTC vs SNAP-GRD comparisons")
    lines.append("")
    lines.append("| comparison | fold mode | probe | metric | RTC mean | SNAP mean | delta RTC-SNAP | RTC wins | SNAP wins | ties |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")

    for row in comparison_rows:
        lines.append(
            f"| {row['comparison']} | "
            f"{row['fold_mode']} | "
            f"{row['probe']} | "
            f"{row['metric']} | "
            f"{row['rtc_mean']} | "
            f"{row['snap_mean']} | "
            f"{row['delta_rtc_minus_snap']} | "
            f"{row['rtc_wins']} | "
            f"{row['snap_wins']} | "
            f"{row['ties']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The primary metric is Average Precision.")
    lines.append("- Threshold-dependent metrics use a threshold selected on the training fold, not the test fold.")
    lines.append("- RTC is favoured only if it improves downstream performance under spatial folds, especially Average Precision and F1.")
    lines.append("- The most important comparisons are:")
    lines.append("  - `s1_rtc_vv_vh` versus `s1_snap_vv_vh` for SAR-only performance.")
    lines.append("  - `s2_s1_rtc_vv_vh` versus `s2_s1_snap_vv_vh` for multimodal performance.")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("After this script, inspect the aggregate and comparison CSVs. Then we can decide whether to run additional fold modes, tune probe settings, or move to embedding consistency and separability analyses.")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------

def default_modalities() -> List[str]:
    return [
        "s2",
        "s1_snap_vv_vh",
        "s1_rtc_vv_vh",
        "s2_s1_snap_vv_vh",
        "s2_s1_rtc_vv_vh",
    ]


def embedding_path_for_modality(
    embedding_dir: Path,
    modality: str,
    stem: str,
) -> Path:
    return embedding_dir / f"croma_embeddings_{modality}_{stem}.npz"


def load_embedding_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        fail(f"Embedding file does not exist: {path_to_str(path)}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_all_embeddings(
    *,
    embedding_dir: Path,
    modalities: Sequence[str],
    stem: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    loaded: Dict[str, Dict[str, np.ndarray]] = {}

    for modality in modalities:
        path = embedding_path_for_modality(embedding_dir, modality, stem)
        arrays = load_embedding_npz(path)
        loaded[modality] = arrays

        x = arrays["embeddings"]
        y = arrays["label_binary"]

        log(
            "OK",
            f"{modality}: loaded embeddings shape={x.shape}, "
            f"positive={int(np.count_nonzero(y == 1))}, "
            f"empty={int(np.count_nonzero(y == 0))}",
        )

    return loaded


def validate_alignment(loaded: Dict[str, Dict[str, np.ndarray]], modalities: Sequence[str]) -> None:
    reference = loaded[modalities[0]]

    for modality in modalities[1:]:
        arrays = loaded[modality]

        for key in ["patch_ids", "label_binary", "cities", "regions"]:
            if not np.array_equal(reference[key], arrays[key]):
                fail(f"Embedding metadata alignment failed for modality={modality}, key={key}")

        if not np.allclose(
            reference["label_positive_percent"].astype(np.float32),
            arrays["label_positive_percent"].astype(np.float32),
            atol=1e-6,
            rtol=0.0,
        ):
            fail(f"label_positive_percent alignment failed for modality={modality}")

    log("OK", "All embedding files are aligned by patch_id, label, city, and region.")


# ---------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------

def build_folds(
    *,
    y: np.ndarray,
    cities: np.ndarray,
    regions: np.ndarray,
    fold_mode: str,
    min_test_samples: int,
    min_train_positive: int,
    min_train_negative: int,
    min_test_positive: int,
    min_test_negative: int,
    max_folds: int,
) -> List[Dict[str, object]]:
    folds: List[Dict[str, object]] = []

    if fold_mode == "leave_one_region":
        groups = sorted(set(str(x) for x in regions))
        group_array = regions.astype(str)
        group_label = "region"

    elif fold_mode == "leave_one_city":
        groups = sorted(set(str(x) for x in cities))
        group_array = cities.astype(str)
        group_label = "city"

    else:
        fail(f"Unsupported fold_mode: {fold_mode}")

    for group in groups:
        test_mask = group_array == group
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        y_train = y[train_idx]
        y_test = y[test_idx]

        train_pos = int(np.count_nonzero(y_train == 1))
        train_neg = int(np.count_nonzero(y_train == 0))
        test_pos = int(np.count_nonzero(y_test == 1))
        test_neg = int(np.count_nonzero(y_test == 0))

        skip_reason = ""

        if len(test_idx) < min_test_samples:
            skip_reason = f"test_samples={len(test_idx)} < {min_test_samples}"
        elif train_pos < min_train_positive:
            skip_reason = f"train_pos={train_pos} < {min_train_positive}"
        elif train_neg < min_train_negative:
            skip_reason = f"train_neg={train_neg} < {min_train_negative}"
        elif test_pos < min_test_positive:
            skip_reason = f"test_pos={test_pos} < {min_test_positive}"
        elif test_neg < min_test_negative:
            skip_reason = f"test_neg={test_neg} < {min_test_negative}"

        folds.append(
            {
                "fold_mode": fold_mode,
                "fold_name": f"{group_label}_{group}",
                "heldout_type": group_label,
                "heldout_value": group,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_samples": int(len(train_idx)),
                "test_samples": int(len(test_idx)),
                "train_positive": train_pos,
                "train_negative": train_neg,
                "test_positive": test_pos,
                "test_negative": test_neg,
                "skip_reason": skip_reason,
            }
        )

    usable = [f for f in folds if f["skip_reason"] == ""]

    if max_folds > 0:
        usable = usable[:max_folds]

    log("OK", f"{fold_mode}: usable folds={len(usable)} / total groups={len(groups)}")

    for f in folds:
        if f["skip_reason"]:
            log("WARN", f"{fold_mode} {f['fold_name']} skipped: {f['skip_reason']}")

    return usable


# ---------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------

def make_probe(probe_name: str, random_state: int):
    if probe_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if probe_name == "linear_svm":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if probe_name == "small_mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128,),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=256,
                        learning_rate_init=1e-3,
                        max_iter=250,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=12,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    fail(f"Unsupported probe: {probe_name}")


def score_model(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return proba[:, 1].astype(np.float32)

    if hasattr(model, "decision_function"):
        decision = model.decision_function(x)
        return np.asarray(decision).astype(np.float32)

    prediction = model.predict(x)
    return np.asarray(prediction).astype(np.float32)


def select_threshold_on_train(
    *,
    y_train: np.ndarray,
    train_scores: np.ndarray,
    n_thresholds: int,
) -> Tuple[float, float]:
    if np.unique(train_scores).size <= 2:
        candidate_thresholds = sorted(set(float(x) for x in train_scores))
    else:
        quantiles = np.linspace(0.02, 0.98, n_thresholds)
        candidate_thresholds = np.unique(np.quantile(train_scores, quantiles)).tolist()

    best_threshold = float(candidate_thresholds[0])
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        pred = (train_scores >= threshold).astype(np.int64)
        score = f1_score(y_train, pred, zero_division=0)

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def safe_metric(metric_name: str, y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        if metric_name == "average_precision":
            return float(average_precision_score(y_true, y_score))

        if metric_name == "roc_auc":
            if len(np.unique(y_true)) < 2:
                return float("nan")
            return float(roc_auc_score(y_true, y_score))

        if metric_name == "f1":
            return float(f1_score(y_true, y_pred, zero_division=0))

        if metric_name == "precision":
            return float(precision_score(y_true, y_pred, zero_division=0))

        if metric_name == "recall":
            return float(recall_score(y_true, y_pred, zero_division=0))

        if metric_name == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))

        if metric_name == "accuracy":
            return float(accuracy_score(y_true, y_pred))

    except Exception:
        return float("nan")

    return float("nan")


def train_and_evaluate_one(
    *,
    x: np.ndarray,
    y: np.ndarray,
    fold: Dict[str, object],
    modality: str,
    probe_name: str,
    random_state: int,
    n_thresholds: int,
) -> Dict[str, object]:
    start_time = time.time()

    train_idx = fold["train_idx"]
    test_idx = fold["test_idx"]

    x_train = x[train_idx]
    y_train = y[train_idx].astype(np.int64)

    x_test = x[test_idx]
    y_test = y[test_idx].astype(np.int64)

    model = make_probe(probe_name, random_state=random_state)

    status = "completed"
    notes = ""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train)

        train_scores = score_model(model, x_train)
        test_scores = score_model(model, x_test)

        threshold, train_best_f1 = select_threshold_on_train(
            y_train=y_train,
            train_scores=train_scores,
            n_thresholds=n_thresholds,
        )

        y_pred = (test_scores >= threshold).astype(np.int64)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        row = {
            "status": status,
            "fold_mode": fold["fold_mode"],
            "fold_name": fold["fold_name"],
            "heldout_type": fold["heldout_type"],
            "heldout_value": fold["heldout_value"],
            "modality": modality,
            "probe": probe_name,

            "train_samples": fold["train_samples"],
            "test_samples": fold["test_samples"],
            "train_positive": fold["train_positive"],
            "train_negative": fold["train_negative"],
            "test_positive": fold["test_positive"],
            "test_negative": fold["test_negative"],

            "threshold": round_float(threshold, 8),
            "train_best_f1_at_threshold": round_float(train_best_f1, 8),

            "average_precision": round_float(safe_metric("average_precision", y_test, test_scores, y_pred), 8),
            "roc_auc": round_float(safe_metric("roc_auc", y_test, test_scores, y_pred), 8),
            "f1": round_float(safe_metric("f1", y_test, test_scores, y_pred), 8),
            "precision": round_float(safe_metric("precision", y_test, test_scores, y_pred), 8),
            "recall": round_float(safe_metric("recall", y_test, test_scores, y_pred), 8),
            "balanced_accuracy": round_float(safe_metric("balanced_accuracy", y_test, test_scores, y_pred), 8),
            "accuracy": round_float(safe_metric("accuracy", y_test, test_scores, y_pred), 8),

            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "test_positive_percent": round_float(100.0 * fold["test_positive"] / fold["test_samples"], 8),
            "pred_positive_percent": round_float(100.0 * int(np.count_nonzero(y_pred == 1)) / len(y_pred), 8),

            "elapsed_seconds": round_float(time.time() - start_time, 3),
            "notes": notes,
        }

        return row

    except Exception as exc:
        return {
            "status": "failed",
            "fold_mode": fold["fold_mode"],
            "fold_name": fold["fold_name"],
            "heldout_type": fold["heldout_type"],
            "heldout_value": fold["heldout_value"],
            "modality": modality,
            "probe": probe_name,

            "train_samples": fold["train_samples"],
            "test_samples": fold["test_samples"],
            "train_positive": fold["train_positive"],
            "train_negative": fold["train_negative"],
            "test_positive": fold["test_positive"],
            "test_negative": fold["test_negative"],

            "threshold": "",
            "train_best_f1_at_threshold": "",

            "average_precision": "",
            "roc_auc": "",
            "f1": "",
            "precision": "",
            "recall": "",
            "balanced_accuracy": "",
            "accuracy": "",

            "tp": "",
            "tn": "",
            "fp": "",
            "fn": "",
            "test_positive_percent": "",
            "pred_positive_percent": "",

            "elapsed_seconds": round_float(time.time() - start_time, 3),
            "notes": repr(exc),
        }


# ---------------------------------------------------------------------
# Aggregation and comparisons
# ---------------------------------------------------------------------

METRIC_NAMES = [
    "average_precision",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "balanced_accuracy",
    "accuracy",
]


def mean_std_median(values: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray([safe_float(v, float("nan")) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    return float(np.mean(arr)), float(np.std(arr)), float(np.median(arr))


def build_aggregate_rows(result_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)

    for row in result_rows:
        if row["status"] != "completed":
            continue

        key = (
            str(row["fold_mode"]),
            str(row["probe"]),
            str(row["modality"]),
        )
        grouped[key].append(row)

    aggregate_rows: List[Dict[str, object]] = []

    for (fold_mode, probe, modality), rows in sorted(grouped.items()):
        out: Dict[str, object] = {
            "fold_mode": fold_mode,
            "probe": probe,
            "modality": modality,
            "n_folds": len(rows),
        }

        for metric in METRIC_NAMES:
            values = [safe_float(r[metric], float("nan")) for r in rows]
            mean, std, median = mean_std_median(values)

            out[f"{metric}_mean"] = round_float(mean, 8)
            out[f"{metric}_std"] = round_float(std, 8)
            out[f"{metric}_median"] = round_float(median, 8)

        aggregate_rows.append(out)

    return aggregate_rows


def build_comparison_rows(result_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    comparisons = [
        {
            "comparison": "sar_only_rtc_vs_snap",
            "rtc_modality": "s1_rtc_vv_vh",
            "snap_modality": "s1_snap_vv_vh",
        },
        {
            "comparison": "joint_rtc_vs_snap",
            "rtc_modality": "s2_s1_rtc_vv_vh",
            "snap_modality": "s2_s1_snap_vv_vh",
        },
    ]

    rows_by_key: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}

    for row in result_rows:
        if row["status"] != "completed":
            continue

        key = (
            str(row["fold_mode"]),
            str(row["probe"]),
            str(row["fold_name"]),
            str(row["modality"]),
        )
        rows_by_key[key] = row

    comparison_rows: List[Dict[str, object]] = []

    fold_modes = sorted(set(str(r["fold_mode"]) for r in result_rows))
    probes = sorted(set(str(r["probe"]) for r in result_rows))

    for comp in comparisons:
        for fold_mode in fold_modes:
            for probe in probes:
                fold_names = sorted(
                    set(
                        str(r["fold_name"])
                        for r in result_rows
                        if r["status"] == "completed"
                        and str(r["fold_mode"]) == fold_mode
                        and str(r["probe"]) == probe
                    )
                )

                for metric in METRIC_NAMES:
                    rtc_values: List[float] = []
                    snap_values: List[float] = []

                    rtc_wins = 0
                    snap_wins = 0
                    ties = 0

                    for fold_name in fold_names:
                        rtc_key = (fold_mode, probe, fold_name, comp["rtc_modality"])
                        snap_key = (fold_mode, probe, fold_name, comp["snap_modality"])

                        if rtc_key not in rows_by_key or snap_key not in rows_by_key:
                            continue

                        rtc_val = safe_float(rows_by_key[rtc_key][metric], float("nan"))
                        snap_val = safe_float(rows_by_key[snap_key][metric], float("nan"))

                        if not np.isfinite(rtc_val) or not np.isfinite(snap_val):
                            continue

                        rtc_values.append(rtc_val)
                        snap_values.append(snap_val)

                        diff = rtc_val - snap_val

                        if abs(diff) <= 1e-12:
                            ties += 1
                        elif diff > 0:
                            rtc_wins += 1
                        else:
                            snap_wins += 1

                    if not rtc_values:
                        continue

                    rtc_mean = float(np.mean(rtc_values))
                    snap_mean = float(np.mean(snap_values))
                    delta = rtc_mean - snap_mean

                    comparison_rows.append(
                        {
                            "comparison": comp["comparison"],
                            "fold_mode": fold_mode,
                            "probe": probe,
                            "metric": metric,
                            "n_matched_folds": len(rtc_values),
                            "rtc_modality": comp["rtc_modality"],
                            "snap_modality": comp["snap_modality"],
                            "rtc_mean": round_float(rtc_mean, 8),
                            "snap_mean": round_float(snap_mean, 8),
                            "delta_rtc_minus_snap": round_float(delta, 8),
                            "rtc_wins": rtc_wins,
                            "snap_wins": snap_wins,
                            "ties": ties,
                        }
                    )

    return comparison_rows


def build_summary(
    *,
    instance_root: Path,
    embedding_dir: Path,
    output_dir: Path,
    result_rows: List[Dict[str, object]],
    aggregate_rows: List[Dict[str, object]],
    comparison_rows: List[Dict[str, object]],
    args: argparse.Namespace,
    started_utc: str,
    output_paths: Dict[str, Path],
) -> Dict[str, object]:
    n_failed = sum(1 for row in result_rows if row["status"] != "completed")

    return {
        "created_utc": started_utc,
        "finished_utc": now_utc(),
        "status": "passed" if n_failed == 0 else "failed",
        "instance_root": path_to_str(instance_root),
        "embedding_dir": path_to_str(embedding_dir),
        "output_dir": path_to_str(output_dir),
        "modalities": list(args.modalities),
        "probes": list(args.probes),
        "fold_modes": list(args.fold_modes),
        "n_result_rows": len(result_rows),
        "n_failed_rows": n_failed,
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "random_state": args.random_state,
            "threshold_candidates": args.threshold_candidates,
            "max_folds": args.max_folds,
            "min_test_samples": args.min_test_samples,
            "min_train_positive": args.min_train_positive,
            "min_train_negative": args.min_train_negative,
            "min_test_positive": args.min_test_positive,
            "min_test_negative": args.min_test_negative,
        },
        "outputs": {key: path_to_str(value) for key, value in output_paths.items()},
        "aggregate_rows": aggregate_rows,
        "comparison_rows": comparison_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frozen probes on CROMA embeddings under spatial folds."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/full_embeddings.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/frozen_probe.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size. Default: 224.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=112,
        help="Stride. Default: 112.",
    )

    parser.add_argument(
        "--edge-mode",
        choices=["cover", "drop"],
        default="cover",
        help="Edge mode. Default: cover.",
    )

    parser.add_argument(
        "--modalities",
        nargs="+",
        default=default_modalities(),
        help="Modalities to evaluate.",
    )

    parser.add_argument(
        "--probes",
        nargs="+",
        default=["logistic_regression", "linear_svm", "small_mlp"],
        choices=["logistic_regression", "linear_svm", "small_mlp"],
        help="Probe models to train.",
    )

    parser.add_argument(
        "--fold-modes",
        nargs="+",
        default=["leave_one_region"],
        choices=["leave_one_region", "leave_one_city"],
        help="Spatial fold modes. Default: leave_one_region.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    parser.add_argument(
        "--threshold-candidates",
        type=int,
        default=101,
        help="Number of candidate thresholds from training scores. Default: 101.",
    )

    parser.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="Debug option. If >0, use only first N folds per fold mode. Default: 0.",
    )

    parser.add_argument(
        "--min-test-samples",
        type=int,
        default=10,
        help="Minimum test samples per fold. Default: 10.",
    )

    parser.add_argument(
        "--min-train-positive",
        type=int,
        default=5,
        help="Minimum positive training samples per fold. Default: 5.",
    )

    parser.add_argument(
        "--min-train-negative",
        type=int,
        default=5,
        help="Minimum negative training samples per fold. Default: 5.",
    )

    parser.add_argument(
        "--min-test-positive",
        type=int,
        default=1,
        help="Minimum positive test samples per fold. Default: 1.",
    )

    parser.add_argument(
        "--min-test-negative",
        type=int,
        default=1,
        help="Minimum negative test samples per fold. Default: 1.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    started_utc = now_utc()

    instance_root: Path = args.instance_root

    embedding_dir: Path = args.embedding_dir or (
        instance_root / "metadata" / "croma_probing" / "full_embeddings"
    )

    output_dir: Path = args.output_dir or (
        instance_root / "metadata" / "croma_probing" / "frozen_probe"
    )

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"
    fold_slug = fold_modes_slug(args.fold_modes)

    result_csv = output_dir / f"frozen_probe_results_{fold_slug}_{stem}.csv"
    aggregate_csv = output_dir / f"frozen_probe_aggregate_{fold_slug}_{stem}.csv"
    comparison_csv = output_dir / f"frozen_probe_rtc_vs_snap_comparison_{fold_slug}_{stem}.csv"
    json_path = output_dir / f"frozen_probe_summary_{fold_slug}_{stem}.json"
    md_path = output_dir / f"frozen_probe_summary_{fold_slug}_{stem}.md"

    output_paths = {
        "result_csv": result_csv,
        "aggregate_csv": aggregate_csv,
        "comparison_csv": comparison_csv,
        "json": json_path,
        "markdown": md_path,
    }

    log("STEP", "Training frozen probes on CROMA embeddings.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Embedding dir: {path_to_str(embedding_dir)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")
    log("INFO", f"Modalities:    {';'.join(args.modalities)}")
    log("INFO", f"Probes:        {';'.join(args.probes)}")
    log("INFO", f"Fold modes:    {';'.join(args.fold_modes)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not embedding_dir.exists():
        fail(f"Embedding dir does not exist: {path_to_str(embedding_dir)}")

    loaded = load_all_embeddings(
        embedding_dir=embedding_dir,
        modalities=args.modalities,
        stem=stem,
    )

    validate_alignment(loaded, args.modalities)

    reference = loaded[args.modalities[0]]
    y = reference["label_binary"].astype(np.int64)
    cities = reference["cities"].astype(str)
    regions = reference["regions"].astype(str)

    all_folds: Dict[str, List[Dict[str, object]]] = {}

    for fold_mode in args.fold_modes:
        all_folds[fold_mode] = build_folds(
            y=y,
            cities=cities,
            regions=regions,
            fold_mode=fold_mode,
            min_test_samples=int(args.min_test_samples),
            min_train_positive=int(args.min_train_positive),
            min_train_negative=int(args.min_train_negative),
            min_test_positive=int(args.min_test_positive),
            min_test_negative=int(args.min_test_negative),
            max_folds=int(args.max_folds),
        )

    result_rows: List[Dict[str, object]] = []

    total_jobs = (
        len(args.modalities)
        * len(args.probes)
        * sum(len(folds) for folds in all_folds.values())
    )

    job_idx = 0

    for fold_mode in args.fold_modes:
        folds = all_folds[fold_mode]

        for probe_name in args.probes:
            for modality in args.modalities:
                x = loaded[modality]["embeddings"].astype(np.float32)

                for fold in folds:
                    job_idx += 1

                    log(
                        "STEP",
                        f"[{job_idx}/{total_jobs}] "
                        f"fold={fold['fold_name']} | probe={probe_name} | modality={modality}",
                    )

                    row = train_and_evaluate_one(
                        x=x,
                        y=y,
                        fold=fold,
                        modality=modality,
                        probe_name=probe_name,
                        random_state=int(args.random_state),
                        n_thresholds=int(args.threshold_candidates),
                    )

                    result_rows.append(row)

                    log(
                        "OK" if row["status"] == "completed" else "ERROR",
                        f"AP={row['average_precision']} | "
                        f"ROC-AUC={row['roc_auc']} | "
                        f"F1={row['f1']} | "
                        f"BalAcc={row['balanced_accuracy']}",
                    )

    aggregate_rows = build_aggregate_rows(result_rows)
    comparison_rows = build_comparison_rows(result_rows)

    summary = build_summary(
        instance_root=instance_root,
        embedding_dir=embedding_dir,
        output_dir=output_dir,
        result_rows=result_rows,
        aggregate_rows=aggregate_rows,
        comparison_rows=comparison_rows,
        args=args,
        started_utc=started_utc,
        output_paths=output_paths,
    )

    log("STEP", "Writing frozen-probe outputs.")

    write_csv(result_csv, result_rows, overwrite=bool(args.overwrite))
    write_csv(aggregate_csv, aggregate_rows, overwrite=bool(args.overwrite))
    write_csv(comparison_csv, comparison_rows, overwrite=bool(args.overwrite))
    write_json(json_path, summary, overwrite=bool(args.overwrite))
    write_markdown(md_path, summary, result_rows, aggregate_rows, comparison_rows, overwrite=bool(args.overwrite))

    log("OK", f"Wrote result CSV:     {path_to_str(result_csv)}")
    log("OK", f"Wrote aggregate CSV:  {path_to_str(aggregate_csv)}")
    log("OK", f"Wrote comparison CSV: {path_to_str(comparison_csv)}")
    log("OK", f"Wrote JSON:           {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:       {path_to_str(md_path)}")

    log("STEP", "Final frozen-probe summary.")
    log("OK" if summary["status"] == "passed" else "ERROR", f"Status: {summary['status']}")
    log("OK", f"Result rows: {summary['n_result_rows']}")
    log("OK", f"Failed rows: {summary['n_failed_rows']}")

    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()