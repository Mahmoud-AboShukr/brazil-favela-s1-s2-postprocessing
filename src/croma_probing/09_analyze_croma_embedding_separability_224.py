#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
09_analyze_croma_embedding_separability_224.py

Criterion 3:
    CROMA embedding separability analysis.

This script analyzes whether CROMA embeddings separate favela-positive patches
from empty/non-favela patches, and whether the embeddings are dominated by
city/region effects.

It uses the full CROMA embedding files created by:

    04_extract_croma_embeddings_full_224.py

Expected inputs:

    <instance-root>/metadata/croma_probing/full_embeddings/
        croma_embeddings_s2_ps224_st112_cover.npz
        croma_embeddings_s1_snap_vv_vh_ps224_st112_cover.npz
        croma_embeddings_s1_rtc_vv_vh_ps224_st112_cover.npz
        croma_embeddings_s2_s1_snap_vv_vh_ps224_st112_cover.npz
        croma_embeddings_s2_s1_rtc_vv_vh_ps224_st112_cover.npz

Main outputs:

    <instance-root>/metadata/croma_probing/criterion3_embedding_separability/
        criterion3_separability_metrics_ps224_st112_cover.csv
        criterion3_knn_scores_ps224_st112_cover.csv
        criterion3_pca_coordinates_ps224_st112_cover.csv
        criterion3_decision_summary_ps224_st112_cover.csv
        criterion3_embedding_separability_summary_ps224_st112_cover.json
        criterion3_embedding_separability_summary_ps224_st112_cover.md

Optional figures:

    figures/criterion3_pca_label_binary_ps224_st112_cover.png
    figures/criterion3_pca_region_ps224_st112_cover.png
    figures/criterion3_pca_city_sample_ps224_st112_cover.png

Optional UMAP, if umap-learn is installed and --run-umap is provided:

    criterion3_umap_coordinates_ps224_st112_cover.csv
    figures/criterion3_umap_label_binary_ps224_st112_cover.png
    figures/criterion3_umap_region_ps224_st112_cover.png

Important:
    This criterion is supporting evidence.
    It should not overrule Criterion 1 downstream frozen-probe performance.

Example:

python src/croma_probing/09_analyze_croma_embedding_separability_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --make-figures `
  --overwrite

Optional UMAP:

python src/croma_probing/09_analyze_croma_embedding_separability_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --make-figures `
  --run-umap `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        silhouette_score,
    )
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "[ERROR] scikit-learn is required.\n"
        "Install it with:\n"
        "    pip install scikit-learn\n\n"
        f"Original error: {exc}"
    )

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


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


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_output_can_be_written(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        fail(
            "Output already exists and --overwrite was not provided:\n"
            f"  {path_to_str(path)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if text == "":
            return default
        out = float(text)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def round_float(value: float, digits: int = 8) -> float:
    return round(safe_float(value, 0.0), digits)


def default_modalities() -> List[str]:
    return [
        "s2",
        "s1_snap_vv_vh",
        "s1_rtc_vv_vh",
        "s2_s1_snap_vv_vh",
        "s2_s1_rtc_vv_vh",
    ]


def pretty_modality(value: str) -> str:
    mapping = {
        "s2": "S2 only",
        "s1_snap_vv_vh": "S1 SNAP-GRD VV/VH",
        "s1_rtc_vv_vh": "S1 RTC VV/VH",
        "s2_s1_snap_vv_vh": "S2 + S1 SNAP-GRD VV/VH",
        "s2_s1_rtc_vv_vh": "S2 + S1 RTC VV/VH",
    }
    return mapping.get(str(value), str(value))


def pretty_comparison(value: str) -> str:
    mapping = {
        "sar_only_rtc_vs_snap": "SAR-only RTC vs SNAP-GRD",
        "joint_rtc_vs_snap": "Joint S2+S1 RTC vs SNAP-GRD",
    }
    return mapping.get(str(value), str(value))


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
            fail(f"No rows to write and no fieldnames were provided: {path_to_str(path)}")
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
    *,
    summary: Dict[str, object],
    metrics_rows: List[Dict[str, object]],
    knn_rows: List[Dict[str, object]],
    decision_rows: List[Dict[str, object]],
    output_paths: Dict[str, Optional[Path]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# Criterion 3 Summary: CROMA Embedding Separability")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Embedding directory: `{summary['embedding_dir']}`")
    lines.append(f"- Output directory: `{summary['output_dir']}`")
    lines.append(f"- Number of patches: `{summary['n_patches']}`")
    lines.append(f"- Embedding dimension: `{summary['embedding_dim']}`")
    lines.append(f"- Modalities: `{';'.join(summary['modalities'])}`")
    lines.append("")

    lines.append("### Main conclusion")
    lines.append("")
    lines.append(summary["main_conclusion"])
    lines.append("")

    lines.append("## Decision summary")
    lines.append("")
    lines.append("| comparison | primary metric | SNAP | RTC | delta RTC-SNAP | label silhouette delta | decision |")
    lines.append("|---|---|---:|---:|---:|---:|---|")

    for row in decision_rows:
        lines.append(
            f"| {row['comparison_pretty']} | "
            f"{row['primary_metric']} | "
            f"{row['snap_primary_value']} | "
            f"{row['rtc_primary_value']} | "
            f"{row['delta_rtc_minus_snap_primary']} | "
            f"{row['delta_rtc_minus_snap_label_silhouette']} | "
            f"{row['decision']} |"
        )

    lines.append("")
    lines.append("## Modality-level separability metrics")
    lines.append("")
    lines.append("| modality | label silhouette | region silhouette | city silhouette | PCA1 var | PCA2 var |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for row in metrics_rows:
        lines.append(
            f"| {row['modality_pretty']} | "
            f"{row['silhouette_label_binary']} | "
            f"{row['silhouette_region']} | "
            f"{row['silhouette_city']} | "
            f"{row['pca_explained_variance_ratio_1']} | "
            f"{row['pca_explained_variance_ratio_2']} |"
        )

    lines.append("")
    lines.append("## kNN probe separability")
    lines.append("")
    lines.append("| modality | AP mean | ROC-AUC mean | F1 mean | balanced accuracy mean |")
    lines.append("|---|---:|---:|---:|---:|")

    for row in knn_rows:
        lines.append(
            f"| {row['modality_pretty']} | "
            f"{row['average_precision_mean']} | "
            f"{row['roc_auc_mean']} | "
            f"{row['f1_mean']} | "
            f"{row['balanced_accuracy_mean']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This criterion evaluates whether the embeddings separate favela-positive patches from empty patches. It uses two complementary ideas. First, silhouette scores measure whether samples with the same label, region, or city are geometrically grouped in embedding space. Second, a simple k-nearest-neighbour classifier checks whether the embedding space is locally informative for the binary patch label.")
    lines.append("")
    lines.append("The label-based metrics indicate favela/non-favela separability. The city and region silhouette scores are diagnostic: high city or region separability can indicate that embeddings contain strong geographic structure, which may or may not be desirable depending on the downstream task.")
    lines.append("")
    lines.append("This criterion is supporting evidence only. The primary decision criterion remains downstream frozen-probe performance from Criterion 1.")
    lines.append("")

    if output_paths.get("pca_label_figure") is not None:
        lines.append("## Optional generated figures")
        lines.append("")
        lines.append(f"- PCA by label: `{path_to_str(output_paths['pca_label_figure'])}`")
        lines.append(f"- PCA by region: `{path_to_str(output_paths['pca_region_figure'])}`")
        lines.append(f"- PCA by city sample: `{path_to_str(output_paths['pca_city_figure'])}`")
        if output_paths.get("umap_label_figure") is not None:
            lines.append(f"- UMAP by label: `{path_to_str(output_paths['umap_label_figure'])}`")
            lines.append(f"- UMAP by region: `{path_to_str(output_paths['umap_region_figure'])}`")
        lines.append("")

    lines.append("## Output files")
    lines.append("")
    for key, value in output_paths.items():
        if value is not None:
            lines.append(f"- `{key}`: `{path_to_str(value)}`")

    lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------

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
    embedding_dir: Path,
    modalities: Sequence[str],
    stem: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    loaded: Dict[str, Dict[str, np.ndarray]] = {}

    for modality in modalities:
        path = embedding_path_for_modality(embedding_dir, modality, stem)
        arrays = load_embedding_npz(path)
        loaded[modality] = arrays

        emb = arrays["embeddings"]

        log(
            "OK",
            f"{modality}: loaded {path_to_str(path)} | shape={emb.shape}",
        )

    return loaded


def validate_alignment(
    loaded: Dict[str, Dict[str, np.ndarray]],
    modalities: Sequence[str],
) -> None:
    ref = loaded[modalities[0]]

    for modality in modalities[1:]:
        cur = loaded[modality]

        for key in [
            "patch_ids",
            "label_binary",
            "cities",
            "regions",
            "label_positive_pixels",
            "label_density_bins",
        ]:
            if not np.array_equal(ref[key], cur[key]):
                fail(f"Metadata alignment failed for modality={modality}, key={key}")

        if not np.allclose(
            ref["label_positive_percent"].astype(np.float32),
            cur["label_positive_percent"].astype(np.float32),
            atol=1e-6,
            rtol=0.0,
        ):
            fail(f"label_positive_percent alignment failed for modality={modality}")

        if ref["embeddings"].shape != cur["embeddings"].shape:
            fail(
                f"Embedding shape mismatch: {modalities[0]}={ref['embeddings'].shape}, "
                f"{modality}={cur['embeddings'].shape}"
            )

    log("OK", "All requested embeddings are aligned by patch, label, city, and region.")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def standardize_embeddings(x: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(x.astype(np.float32)).astype(np.float32)


def maybe_sample_indices(
    n: int,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    if sample_size <= 0 or sample_size >= n:
        return np.arange(n)

    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=sample_size, replace=False))


def safe_silhouette(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    random_state: int,
) -> float:
    labels = np.asarray(labels)

    if len(np.unique(labels)) < 2:
        return float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sample_size > 0 and sample_size < x.shape[0]:
                return float(
                    silhouette_score(
                        x,
                        labels,
                        metric="euclidean",
                        sample_size=sample_size,
                        random_state=random_state,
                    )
                )
            return float(silhouette_score(x, labels, metric="euclidean"))
    except Exception:
        return float("nan")


def fit_pca_coordinates(
    x_std: np.ndarray,
    *,
    n_components: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_components = max(2, int(n_components))
    n_components = min(n_components, x_std.shape[1])

    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(x_std)

    return coords.astype(np.float32), pca.explained_variance_ratio_.astype(np.float32)


def build_pca_rows(
    *,
    modality: str,
    metadata: Dict[str, np.ndarray],
    coords: np.ndarray,
) -> List[Dict[str, object]]:
    patch_ids = metadata["patch_ids"].astype(str)
    cities = metadata["cities"].astype(str)
    regions = metadata["regions"].astype(str)
    labels = metadata["label_binary"].astype(np.int64)
    label_positive_percent = metadata["label_positive_percent"].astype(np.float32)
    density_bins = metadata["label_density_bins"].astype(str)

    rows: List[Dict[str, object]] = []

    for i in range(coords.shape[0]):
        rows.append(
            {
                "modality": modality,
                "modality_pretty": pretty_modality(modality),
                "patch_index": i,
                "patch_id": patch_ids[i],
                "city": cities[i],
                "region": regions[i],
                "label_binary": int(labels[i]),
                "label_positive_percent": round_float(float(label_positive_percent[i]), 8),
                "label_density_bin": density_bins[i],
                "pca_1": round_float(float(coords[i, 0]), 8),
                "pca_2": round_float(float(coords[i, 1]), 8),
            }
        )

    return rows


def run_knn_separability(
    *,
    x: np.ndarray,
    y: np.ndarray,
    modality: str,
    random_state: int,
    n_splits: int,
    test_size: float,
    n_neighbors: int,
) -> Dict[str, object]:
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )

    metric_values: Dict[str, List[float]] = defaultdict(list)

    for split_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x[train_idx]
        y_train = y[train_idx].astype(np.int64)
        x_test = x[test_idx]
        y_test = y[test_idx].astype(np.int64)

        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=int(n_neighbors),
                        weights="distance",
                        metric="minkowski",
                        p=2,
                    ),
                ),
            ]
        )

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(x_test)[:, 1]
        else:
            y_score = y_pred.astype(np.float32)

        metric_values["average_precision"].append(float(average_precision_score(y_test, y_score)))

        if len(np.unique(y_test)) >= 2:
            metric_values["roc_auc"].append(float(roc_auc_score(y_test, y_score)))
        else:
            metric_values["roc_auc"].append(float("nan"))

        metric_values["f1"].append(float(f1_score(y_test, y_pred, zero_division=0)))
        metric_values["precision"].append(float(precision_score(y_test, y_pred, zero_division=0)))
        metric_values["recall"].append(float(recall_score(y_test, y_pred, zero_division=0)))
        metric_values["balanced_accuracy"].append(float(balanced_accuracy_score(y_test, y_pred)))
        metric_values["accuracy"].append(float(accuracy_score(y_test, y_pred)))

    row: Dict[str, object] = {
        "modality": modality,
        "modality_pretty": pretty_modality(modality),
        "n_splits": int(n_splits),
        "test_size": float(test_size),
        "n_neighbors": int(n_neighbors),
    }

    for metric_name, values in metric_values.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            row[f"{metric_name}_mean"] = ""
            row[f"{metric_name}_std"] = ""
            row[f"{metric_name}_median"] = ""
        else:
            row[f"{metric_name}_mean"] = round_float(float(np.mean(arr)), 8)
            row[f"{metric_name}_std"] = round_float(float(np.std(arr)), 8)
            row[f"{metric_name}_median"] = round_float(float(np.median(arr)), 8)

    return row


def analyze_one_modality(
    *,
    modality: str,
    arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]], np.ndarray]:
    log("STEP", f"Analyzing separability for modality: {modality}")

    x = arrays["embeddings"].astype(np.float32)
    y = arrays["label_binary"].astype(np.int64)
    cities = arrays["cities"].astype(str)
    regions = arrays["regions"].astype(str)

    x_std = standardize_embeddings(x)

    label_silhouette = safe_silhouette(
        x_std,
        y,
        sample_size=int(args.silhouette_sample_size),
        random_state=int(args.random_state),
    )

    region_silhouette = safe_silhouette(
        x_std,
        regions,
        sample_size=int(args.silhouette_sample_size),
        random_state=int(args.random_state),
    )

    city_silhouette = safe_silhouette(
        x_std,
        cities,
        sample_size=int(args.silhouette_sample_size),
        random_state=int(args.random_state),
    )

    pca_coords, pca_var = fit_pca_coordinates(
        x_std,
        n_components=int(args.pca_components),
        random_state=int(args.random_state),
    )

    pca_rows = build_pca_rows(
        modality=modality,
        metadata=arrays,
        coords=pca_coords,
    )

    knn_row = run_knn_separability(
        x=x,
        y=y,
        modality=modality,
        random_state=int(args.random_state),
        n_splits=int(args.knn_splits),
        test_size=float(args.knn_test_size),
        n_neighbors=int(args.knn_neighbors),
    )

    metrics_row = {
        "modality": modality,
        "modality_pretty": pretty_modality(modality),
        "n_patches": int(x.shape[0]),
        "embedding_dim": int(x.shape[1]),
        "positive_count": int(np.count_nonzero(y == 1)),
        "empty_count": int(np.count_nonzero(y == 0)),
        "silhouette_label_binary": round_float(label_silhouette, 8),
        "silhouette_region": round_float(region_silhouette, 8),
        "silhouette_city": round_float(city_silhouette, 8),
        "pca_explained_variance_ratio_1": round_float(float(pca_var[0]), 8),
        "pca_explained_variance_ratio_2": round_float(float(pca_var[1]), 8),
        "pca_explained_variance_ratio_first_5": round_float(float(np.sum(pca_var[:5])), 8) if len(pca_var) >= 5 else "",
        "pca_explained_variance_ratio_first_10": round_float(float(np.sum(pca_var[:10])), 8) if len(pca_var) >= 10 else "",
        "silhouette_sample_size": int(args.silhouette_sample_size),
    }

    log(
        "OK",
        f"{modality}: label_sil={metrics_row['silhouette_label_binary']}, "
        f"kNN_AP={knn_row['average_precision_mean']}",
    )

    return metrics_row, knn_row, pca_rows, pca_coords


# ---------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------

def build_umap_rows(
    *,
    modality: str,
    metadata: Dict[str, np.ndarray],
    coords: np.ndarray,
    sample_indices: np.ndarray,
) -> List[Dict[str, object]]:
    patch_ids = metadata["patch_ids"].astype(str)
    cities = metadata["cities"].astype(str)
    regions = metadata["regions"].astype(str)
    labels = metadata["label_binary"].astype(np.int64)
    label_positive_percent = metadata["label_positive_percent"].astype(np.float32)
    density_bins = metadata["label_density_bins"].astype(str)

    rows: List[Dict[str, object]] = []

    for local_i, original_i in enumerate(sample_indices):
        rows.append(
            {
                "modality": modality,
                "modality_pretty": pretty_modality(modality),
                "patch_index": int(original_i),
                "patch_id": patch_ids[original_i],
                "city": cities[original_i],
                "region": regions[original_i],
                "label_binary": int(labels[original_i]),
                "label_positive_percent": round_float(float(label_positive_percent[original_i]), 8),
                "label_density_bin": density_bins[original_i],
                "umap_1": round_float(float(coords[local_i, 0]), 8),
                "umap_2": round_float(float(coords[local_i, 1]), 8),
            }
        )

    return rows


def compute_umap_for_modalities(
    *,
    loaded: Dict[str, Dict[str, np.ndarray]],
    modalities: Sequence[str],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    if not HAS_UMAP:
        log("WARN", "umap-learn is not installed. Skipping UMAP.")
        return []

    all_rows: List[Dict[str, object]] = []

    for modality in modalities:
        log("STEP", f"Computing UMAP for modality: {modality}")

        arrays = loaded[modality]
        x = arrays["embeddings"].astype(np.float32)
        x_std = standardize_embeddings(x)

        sample_indices = maybe_sample_indices(
            n=x_std.shape[0],
            sample_size=int(args.umap_sample_size),
            random_state=int(args.random_state),
        )

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(args.umap_neighbors),
            min_dist=float(args.umap_min_dist),
            metric="euclidean",
            random_state=int(args.random_state),
        )

        coords = reducer.fit_transform(x_std[sample_indices]).astype(np.float32)

        rows = build_umap_rows(
            modality=modality,
            metadata=arrays,
            coords=coords,
            sample_indices=sample_indices,
        )

        all_rows.extend(rows)

        log("OK", f"{modality}: UMAP rows={len(rows)}")

    return all_rows


# ---------------------------------------------------------------------
# Decision summary
# ---------------------------------------------------------------------

def value_from_rows(
    rows: List[Dict[str, object]],
    modality: str,
    key: str,
) -> float:
    for row in rows:
        if row["modality"] == modality:
            return safe_float(row.get(key, ""))
    return float("nan")


def build_decision_rows(
    *,
    metrics_rows: List[Dict[str, object]],
    knn_rows: List[Dict[str, object]],
    min_delta: float,
) -> List[Dict[str, object]]:
    comparisons = [
        {
            "comparison": "sar_only_rtc_vs_snap",
            "snap_modality": "s1_snap_vv_vh",
            "rtc_modality": "s1_rtc_vv_vh",
        },
        {
            "comparison": "joint_rtc_vs_snap",
            "snap_modality": "s2_s1_snap_vv_vh",
            "rtc_modality": "s2_s1_rtc_vv_vh",
        },
    ]

    rows: List[Dict[str, object]] = []

    for comp in comparisons:
        snap = comp["snap_modality"]
        rtc = comp["rtc_modality"]

        snap_primary = value_from_rows(knn_rows, snap, "average_precision_mean")
        rtc_primary = value_from_rows(knn_rows, rtc, "average_precision_mean")
        delta_primary = rtc_primary - snap_primary

        snap_label_sil = value_from_rows(metrics_rows, snap, "silhouette_label_binary")
        rtc_label_sil = value_from_rows(metrics_rows, rtc, "silhouette_label_binary")
        delta_label_sil = rtc_label_sil - snap_label_sil

        snap_region_sil = value_from_rows(metrics_rows, snap, "silhouette_region")
        rtc_region_sil = value_from_rows(metrics_rows, rtc, "silhouette_region")
        delta_region_sil = rtc_region_sil - snap_region_sil

        snap_city_sil = value_from_rows(metrics_rows, snap, "silhouette_city")
        rtc_city_sil = value_from_rows(metrics_rows, rtc, "silhouette_city")
        delta_city_sil = rtc_city_sil - snap_city_sil

        if delta_primary > min_delta:
            decision = "RTC stronger separability"
        elif delta_primary < -min_delta:
            decision = "SNAP-GRD stronger separability"
        else:
            if delta_label_sil > min_delta:
                decision = "Near tie by kNN, RTC higher label silhouette"
            elif delta_label_sil < -min_delta:
                decision = "Near tie by kNN, SNAP-GRD higher label silhouette"
            else:
                decision = "Near tie / mixed"

        rows.append(
            {
                "comparison": comp["comparison"],
                "comparison_pretty": pretty_comparison(comp["comparison"]),
                "primary_metric": "kNN_average_precision_mean",
                "snap_modality": snap,
                "snap_modality_pretty": pretty_modality(snap),
                "rtc_modality": rtc,
                "rtc_modality_pretty": pretty_modality(rtc),
                "snap_primary_value": round_float(snap_primary, 8),
                "rtc_primary_value": round_float(rtc_primary, 8),
                "delta_rtc_minus_snap_primary": round_float(delta_primary, 8),
                "snap_label_silhouette": round_float(snap_label_sil, 8),
                "rtc_label_silhouette": round_float(rtc_label_sil, 8),
                "delta_rtc_minus_snap_label_silhouette": round_float(delta_label_sil, 8),
                "snap_region_silhouette": round_float(snap_region_sil, 8),
                "rtc_region_silhouette": round_float(rtc_region_sil, 8),
                "delta_rtc_minus_snap_region_silhouette": round_float(delta_region_sil, 8),
                "snap_city_silhouette": round_float(snap_city_sil, 8),
                "rtc_city_silhouette": round_float(rtc_city_sil, 8),
                "delta_rtc_minus_snap_city_silhouette": round_float(delta_city_sil, 8),
                "decision": decision,
            }
        )

    return rows


def build_main_conclusion(decision_rows: List[Dict[str, object]]) -> str:
    decision_by_comp = {
        row["comparison"]: row["decision"]
        for row in decision_rows
    }

    sar_decision = decision_by_comp.get("sar_only_rtc_vs_snap", "")
    joint_decision = decision_by_comp.get("joint_rtc_vs_snap", "")

    if "SNAP-GRD" in sar_decision and "SNAP-GRD" in joint_decision:
        return (
            "Criterion 3 supports SNAP-GRD: both the SAR-only and joint S2+S1 "
            "comparisons show stronger local label separability for SNAP-GRD embeddings."
        )

    if "RTC" in sar_decision and "RTC" in joint_decision:
        return (
            "Criterion 3 supports RTC: both the SAR-only and joint S2+S1 "
            "comparisons show stronger local label separability for RTC embeddings."
        )

    if "SNAP-GRD" in joint_decision:
        return (
            "Criterion 3 leans toward SNAP-GRD in the joint S2+S1 setting, while the "
            "SAR-only result is mixed or favours RTC. Because the final representation is "
            "expected to use optical-radar information, the joint result should be considered important."
        )

    if "RTC" in joint_decision:
        return (
            "Criterion 3 leans toward RTC in the joint S2+S1 setting, while the SAR-only "
            "result is mixed or favours SNAP-GRD."
        )

    return (
        "Criterion 3 is mixed or near-tied. Embedding separability does not provide a strong "
        "standalone preference between RTC and SNAP-GRD."
    )


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def plot_embedding_grid(
    *,
    rows: List[Dict[str, object]],
    modalities: Sequence[str],
    x_key: str,
    y_key: str,
    color_key: str,
    title: str,
    output_path: Path,
    overwrite: bool,
    sample_per_modality: int,
    random_state: int,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        log("WARN", "matplotlib is not installed; skipping figure.")
        return None

    ensure_output_can_be_written(output_path, overwrite)

    rng = np.random.default_rng(random_state)

    rows_by_modality: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for row in rows:
        rows_by_modality[str(row["modality"])].append(row)

    n = len(modalities)
    ncols = 2
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 5 * nrows))
    axes_arr = np.asarray(axes).reshape(-1)

    for ax_idx, modality in enumerate(modalities):
        ax = axes_arr[ax_idx]

        modality_rows = rows_by_modality.get(modality, [])

        if sample_per_modality > 0 and len(modality_rows) > sample_per_modality:
            idx = rng.choice(len(modality_rows), size=sample_per_modality, replace=False)
            modality_rows = [modality_rows[int(i)] for i in idx]

        x = np.asarray([safe_float(row[x_key]) for row in modality_rows], dtype=np.float32)
        y = np.asarray([safe_float(row[y_key]) for row in modality_rows], dtype=np.float32)
        c = np.asarray([str(row[color_key]) for row in modality_rows])

        unique_values = sorted(set(c.tolist()))

        for value in unique_values:
            mask = c == value
            ax.scatter(x[mask], y[mask], s=4, alpha=0.55, label=str(value))

        ax.set_title(pretty_modality(modality))
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)

        if len(unique_values) <= 8:
            ax.legend(markerscale=3, fontsize=8)

        ax.grid(alpha=0.2)

    for j in range(len(modalities), len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------
# Summary payload
# ---------------------------------------------------------------------

def build_summary_payload(
    *,
    instance_root: Path,
    embedding_dir: Path,
    output_dir: Path,
    modalities: Sequence[str],
    n_patches: int,
    embedding_dim: int,
    metrics_rows: List[Dict[str, object]],
    knn_rows: List[Dict[str, object]],
    decision_rows: List[Dict[str, object]],
    main_conclusion: str,
    output_paths: Dict[str, Optional[Path]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "created_utc": now_utc(),
        "status": "passed",
        "instance_root": path_to_str(instance_root),
        "embedding_dir": path_to_str(embedding_dir),
        "output_dir": path_to_str(output_dir),
        "modalities": list(modalities),
        "n_patches": int(n_patches),
        "embedding_dim": int(embedding_dim),
        "main_conclusion": main_conclusion,
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "random_state": args.random_state,
            "silhouette_sample_size": args.silhouette_sample_size,
            "pca_components": args.pca_components,
            "knn_splits": args.knn_splits,
            "knn_test_size": args.knn_test_size,
            "knn_neighbors": args.knn_neighbors,
            "run_umap": bool(args.run_umap),
            "umap_sample_size": args.umap_sample_size,
            "make_figures": bool(args.make_figures),
        },
        "outputs": {
            key: "" if value is None else path_to_str(value)
            for key, value in output_paths.items()
        },
        "decision_rows": decision_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CROMA embedding separability for Criterion 3."
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
        help="Default: <instance-root>/metadata/croma_probing/criterion3_embedding_separability.",
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
        help="Modalities to analyze.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=5000,
        help="Sample size for silhouette scores. Default: 5000.",
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help="Number of PCA components. Default: 10.",
    )

    parser.add_argument(
        "--knn-splits",
        type=int,
        default=5,
        help="Number of repeated stratified kNN splits. Default: 5.",
    )

    parser.add_argument(
        "--knn-test-size",
        type=float,
        default=0.25,
        help="Test size for kNN separability probe. Default: 0.25.",
    )

    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=15,
        help="Number of neighbours for kNN. Default: 15.",
    )

    parser.add_argument(
        "--min-decision-delta",
        type=float,
        default=1e-4,
        help="Minimum delta to declare one variant stronger. Default: 1e-4.",
    )

    parser.add_argument(
        "--make-figures",
        action="store_true",
        help="Generate PCA/UMAP figures if plotting libraries are available.",
    )

    parser.add_argument(
        "--figure-sample-per-modality",
        type=int,
        default=3000,
        help="Maximum plotted points per modality. Default: 3000.",
    )

    parser.add_argument(
        "--run-umap",
        action="store_true",
        help="Run optional UMAP if umap-learn is installed.",
    )

    parser.add_argument(
        "--umap-sample-size",
        type=int,
        default=4000,
        help="Sample size per modality for UMAP. Default: 4000.",
    )

    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors. Default: 30.",
    )

    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist. Default: 0.1.",
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

    instance_root: Path = args.instance_root

    embedding_dir: Path = args.embedding_dir or (
        instance_root / "metadata" / "croma_probing" / "full_embeddings"
    )

    output_dir: Path = args.output_dir or (
        instance_root / "metadata" / "croma_probing" / "criterion3_embedding_separability"
    )

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    metrics_csv = output_dir / f"criterion3_separability_metrics_{stem}.csv"
    knn_csv = output_dir / f"criterion3_knn_scores_{stem}.csv"
    pca_csv = output_dir / f"criterion3_pca_coordinates_{stem}.csv"
    umap_csv = output_dir / f"criterion3_umap_coordinates_{stem}.csv"
    decision_csv = output_dir / f"criterion3_decision_summary_{stem}.csv"
    json_path = output_dir / f"criterion3_embedding_separability_summary_{stem}.json"
    md_path = output_dir / f"criterion3_embedding_separability_summary_{stem}.md"

    figure_dir = output_dir / "figures"

    pca_label_figure: Optional[Path] = None
    pca_region_figure: Optional[Path] = None
    pca_city_figure: Optional[Path] = None
    umap_label_figure: Optional[Path] = None
    umap_region_figure: Optional[Path] = None

    if args.make_figures:
        pca_label_figure = figure_dir / f"criterion3_pca_label_binary_{stem}.png"
        pca_region_figure = figure_dir / f"criterion3_pca_region_{stem}.png"
        pca_city_figure = figure_dir / f"criterion3_pca_city_sample_{stem}.png"

        if args.run_umap:
            umap_label_figure = figure_dir / f"criterion3_umap_label_binary_{stem}.png"
            umap_region_figure = figure_dir / f"criterion3_umap_region_{stem}.png"

    output_paths: Dict[str, Optional[Path]] = {
        "metrics_csv": metrics_csv,
        "knn_csv": knn_csv,
        "pca_csv": pca_csv,
        "umap_csv": umap_csv if args.run_umap else None,
        "decision_csv": decision_csv,
        "json": json_path,
        "markdown": md_path,
        "pca_label_figure": pca_label_figure,
        "pca_region_figure": pca_region_figure,
        "pca_city_figure": pca_city_figure,
        "umap_label_figure": umap_label_figure,
        "umap_region_figure": umap_region_figure,
    }

    log("STEP", "Running Criterion 3 CROMA embedding separability analysis.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Embedding dir: {path_to_str(embedding_dir)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")
    log("INFO", f"Modalities:    {';'.join(args.modalities)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not embedding_dir.exists():
        fail(f"Embedding directory does not exist: {path_to_str(embedding_dir)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_all_embeddings(
        embedding_dir=embedding_dir,
        modalities=args.modalities,
        stem=stem,
    )

    validate_alignment(loaded, args.modalities)

    reference = loaded[args.modalities[0]]
    n_patches = int(reference["embeddings"].shape[0])
    embedding_dim = int(reference["embeddings"].shape[1])

    metrics_rows: List[Dict[str, object]] = []
    knn_rows: List[Dict[str, object]] = []
    all_pca_rows: List[Dict[str, object]] = []

    pca_coords_by_modality: Dict[str, np.ndarray] = {}

    for modality in args.modalities:
        metrics_row, knn_row, pca_rows, pca_coords = analyze_one_modality(
            modality=modality,
            arrays=loaded[modality],
            args=args,
        )

        metrics_rows.append(metrics_row)
        knn_rows.append(knn_row)
        all_pca_rows.extend(pca_rows)
        pca_coords_by_modality[modality] = pca_coords

    decision_rows = build_decision_rows(
        metrics_rows=metrics_rows,
        knn_rows=knn_rows,
        min_delta=float(args.min_decision_delta),
    )

    main_conclusion = build_main_conclusion(decision_rows)

    umap_rows: List[Dict[str, object]] = []

    if args.run_umap:
        if not HAS_UMAP:
            log("WARN", "UMAP requested but umap-learn is not installed. Skipping UMAP.")
        else:
            umap_rows = compute_umap_for_modalities(
                loaded=loaded,
                modalities=args.modalities,
                args=args,
            )

    if args.make_figures:
        output_paths["pca_label_figure"] = plot_embedding_grid(
            rows=all_pca_rows,
            modalities=args.modalities,
            x_key="pca_1",
            y_key="pca_2",
            color_key="label_binary",
            title="PCA projection coloured by binary label",
            output_path=pca_label_figure,
            overwrite=bool(args.overwrite),
            sample_per_modality=int(args.figure_sample_per_modality),
            random_state=int(args.random_state),
        )

        output_paths["pca_region_figure"] = plot_embedding_grid(
            rows=all_pca_rows,
            modalities=args.modalities,
            x_key="pca_1",
            y_key="pca_2",
            color_key="region",
            title="PCA projection coloured by region",
            output_path=pca_region_figure,
            overwrite=bool(args.overwrite),
            sample_per_modality=int(args.figure_sample_per_modality),
            random_state=int(args.random_state),
        )

        output_paths["pca_city_figure"] = plot_embedding_grid(
            rows=all_pca_rows,
            modalities=args.modalities,
            x_key="pca_1",
            y_key="pca_2",
            color_key="city",
            title="PCA projection coloured by city",
            output_path=pca_city_figure,
            overwrite=bool(args.overwrite),
            sample_per_modality=int(args.figure_sample_per_modality),
            random_state=int(args.random_state),
        )

        if args.run_umap and umap_rows:
            output_paths["umap_label_figure"] = plot_embedding_grid(
                rows=umap_rows,
                modalities=args.modalities,
                x_key="umap_1",
                y_key="umap_2",
                color_key="label_binary",
                title="UMAP projection coloured by binary label",
                output_path=umap_label_figure,
                overwrite=bool(args.overwrite),
                sample_per_modality=int(args.figure_sample_per_modality),
                random_state=int(args.random_state),
            )

            output_paths["umap_region_figure"] = plot_embedding_grid(
                rows=umap_rows,
                modalities=args.modalities,
                x_key="umap_1",
                y_key="umap_2",
                color_key="region",
                title="UMAP projection coloured by region",
                output_path=umap_region_figure,
                overwrite=bool(args.overwrite),
                sample_per_modality=int(args.figure_sample_per_modality),
                random_state=int(args.random_state),
            )

    summary_payload = build_summary_payload(
        instance_root=instance_root,
        embedding_dir=embedding_dir,
        output_dir=output_dir,
        modalities=args.modalities,
        n_patches=n_patches,
        embedding_dim=embedding_dim,
        metrics_rows=metrics_rows,
        knn_rows=knn_rows,
        decision_rows=decision_rows,
        main_conclusion=main_conclusion,
        output_paths=output_paths,
        args=args,
    )

    log("STEP", "Writing Criterion 3 outputs.")

    write_csv(
        metrics_csv,
        metrics_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "modality",
            "modality_pretty",
            "n_patches",
            "embedding_dim",
            "positive_count",
            "empty_count",
            "silhouette_label_binary",
            "silhouette_region",
            "silhouette_city",
            "pca_explained_variance_ratio_1",
            "pca_explained_variance_ratio_2",
            "pca_explained_variance_ratio_first_5",
            "pca_explained_variance_ratio_first_10",
            "silhouette_sample_size",
        ],
    )

    write_csv(
        knn_csv,
        knn_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "modality",
            "modality_pretty",
            "n_splits",
            "test_size",
            "n_neighbors",
            "average_precision_mean",
            "average_precision_std",
            "average_precision_median",
            "roc_auc_mean",
            "roc_auc_std",
            "roc_auc_median",
            "f1_mean",
            "f1_std",
            "f1_median",
            "precision_mean",
            "precision_std",
            "precision_median",
            "recall_mean",
            "recall_std",
            "recall_median",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "balanced_accuracy_median",
            "accuracy_mean",
            "accuracy_std",
            "accuracy_median",
        ],
    )

    write_csv(
        pca_csv,
        all_pca_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "modality",
            "modality_pretty",
            "patch_index",
            "patch_id",
            "city",
            "region",
            "label_binary",
            "label_positive_percent",
            "label_density_bin",
            "pca_1",
            "pca_2",
        ],
    )

    if args.run_umap and umap_rows:
        write_csv(
            umap_csv,
            umap_rows,
            overwrite=bool(args.overwrite),
            fieldnames=[
                "modality",
                "modality_pretty",
                "patch_index",
                "patch_id",
                "city",
                "region",
                "label_binary",
                "label_positive_percent",
                "label_density_bin",
                "umap_1",
                "umap_2",
            ],
        )

    write_csv(
        decision_csv,
        decision_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "comparison",
            "comparison_pretty",
            "primary_metric",
            "snap_modality",
            "snap_modality_pretty",
            "rtc_modality",
            "rtc_modality_pretty",
            "snap_primary_value",
            "rtc_primary_value",
            "delta_rtc_minus_snap_primary",
            "snap_label_silhouette",
            "rtc_label_silhouette",
            "delta_rtc_minus_snap_label_silhouette",
            "snap_region_silhouette",
            "rtc_region_silhouette",
            "delta_rtc_minus_snap_region_silhouette",
            "snap_city_silhouette",
            "rtc_city_silhouette",
            "delta_rtc_minus_snap_city_silhouette",
            "decision",
        ],
    )

    write_json(json_path, summary_payload, overwrite=bool(args.overwrite))

    write_markdown(
        md_path,
        summary=summary_payload,
        metrics_rows=metrics_rows,
        knn_rows=knn_rows,
        decision_rows=decision_rows,
        output_paths=output_paths,
        overwrite=bool(args.overwrite),
    )

    log("OK", f"Wrote metrics CSV:  {path_to_str(metrics_csv)}")
    log("OK", f"Wrote kNN CSV:      {path_to_str(knn_csv)}")
    log("OK", f"Wrote PCA CSV:      {path_to_str(pca_csv)}")

    if args.run_umap and umap_rows:
        log("OK", f"Wrote UMAP CSV:     {path_to_str(umap_csv)}")

    log("OK", f"Wrote decision CSV: {path_to_str(decision_csv)}")
    log("OK", f"Wrote JSON:         {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:     {path_to_str(md_path)}")

    if output_paths["pca_label_figure"] is not None:
        log("OK", f"Wrote PCA label figure:  {path_to_str(output_paths['pca_label_figure'])}")

    if output_paths["pca_region_figure"] is not None:
        log("OK", f"Wrote PCA region figure: {path_to_str(output_paths['pca_region_figure'])}")

    if output_paths["pca_city_figure"] is not None:
        log("OK", f"Wrote PCA city figure:   {path_to_str(output_paths['pca_city_figure'])}")

    if output_paths["umap_label_figure"] is not None:
        log("OK", f"Wrote UMAP label figure: {path_to_str(output_paths['umap_label_figure'])}")

    if output_paths["umap_region_figure"] is not None:
        log("OK", f"Wrote UMAP region figure:{path_to_str(output_paths['umap_region_figure'])}")

    log("STEP", "Final Criterion 3 summary.")
    log("OK", "Status: passed")
    log("OK", f"Patches: {n_patches}")
    log("OK", f"Embedding dimension: {embedding_dim}")
    log("OK", f"Main conclusion: {main_conclusion}")


if __name__ == "__main__":
    main()