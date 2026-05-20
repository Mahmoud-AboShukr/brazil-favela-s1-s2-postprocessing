#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_validate_croma_embeddings_224.py

Validate full CROMA embedding files before frozen-probe experiments.

This script checks that the five full embedding files produced by:

    04_extract_croma_embeddings_full_224.py

are complete, aligned, finite, non-degenerate, and ready for downstream probing.

Expected embedding files:

    croma_embeddings_s2_ps224_st112_cover.npz
    croma_embeddings_s1_snap_vv_vh_ps224_st112_cover.npz
    croma_embeddings_s1_rtc_vv_vh_ps224_st112_cover.npz
    croma_embeddings_s2_s1_snap_vv_vh_ps224_st112_cover.npz
    croma_embeddings_s2_s1_rtc_vv_vh_ps224_st112_cover.npz

Main checks:

    1. All expected files exist.
    2. Each embedding array has shape (12699, 768).
    3. Required metadata arrays exist.
    4. patch_ids are unique.
    5. patch_ids are identical and in the same order across modalities.
    6. labels, cities, and regions are identical across modalities.
    7. embeddings contain no NaN/Inf.
    8. embeddings have non-zero variance.
    9. SNAP and RTC embeddings are not accidentally identical.
    10. label counts match the validated patch metadata:
        positive = 6382
        empty = 6317

Outputs:

    <instance-root>/metadata/croma_probing/full_embeddings/
        croma_embedding_validation_summary_ps224_st112_cover.csv
        croma_embedding_validation_checks_ps224_st112_cover.csv
        croma_embedding_validation_pairwise_ps224_st112_cover.csv
        croma_embedding_validation_ps224_st112_cover.json
        croma_embedding_validation_ps224_st112_cover.md

Example:

python src/croma_probing/05_validate_croma_embeddings_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


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


def safe_int(value: object, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def normalize_scalar_array_value(value: object) -> str:
    arr = np.asarray(value)

    if arr.size == 0:
        return ""

    return str(arr.reshape(-1)[0])


def round_float(value: float, digits: int = 8) -> float:
    if value is None:
        return 0.0

    value = float(value)

    if math.isnan(value) or math.isinf(value):
        return 0.0

    return round(value, digits)


# ---------------------------------------------------------------------
# I/O
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
            fail(f"No rows and no fieldnames for CSV: {path_to_str(path)}")
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
    modality_rows: List[Dict[str, object]],
    check_rows: List[Dict[str, object]],
    pairwise_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# CROMA embedding validation")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Embedding dir: `{summary['embedding_dir']}`")
    lines.append(f"- Modalities checked: `{';'.join(summary['modalities'])}`")
    lines.append(f"- Expected rows per modality: `{summary['expected_rows']}`")
    lines.append(f"- Expected embedding dimension: `{summary['expected_embedding_dim']}`")
    lines.append(f"- Error count: `{summary['error_count']}`")
    lines.append(f"- Warning count: `{summary['warning_count']}`")
    lines.append("")

    lines.append("## Modality-level validation")
    lines.append("")
    lines.append(
        "| modality | status | rows | dim | positive | empty | finite | global std | zero-var dims | file size MB |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|---:|")

    for row in modality_rows:
        lines.append(
            f"| {row['modality']} | "
            f"{row['status']} | "
            f"{row['n_rows']} | "
            f"{row['embedding_dim']} | "
            f"{row['positive_count']} | "
            f"{row['empty_count']} | "
            f"{row['all_finite']} | "
            f"{row['global_std']} | "
            f"{row['zero_variance_dimensions']} | "
            f"{row['file_size_mb']} |"
        )

    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| check | severity | status | details |")
    lines.append("|---|---|---|---|")

    for row in check_rows:
        lines.append(
            f"| {row['check_name']} | "
            f"{row['severity']} | "
            f"{row['status']} | "
            f"{row['details']} |"
        )

    lines.append("")
    lines.append("## Pairwise embedding checks")
    lines.append("")
    lines.append(
        "| modality A | modality B | same shape | max abs diff | mean abs diff | mean cosine | identical? | status |"
    )
    lines.append("|---|---|---|---:|---:|---:|---|---|")

    for row in pairwise_rows:
        lines.append(
            f"| {row['modality_a']} | "
            f"{row['modality_b']} | "
            f"{row['same_shape']} | "
            f"{row['max_abs_diff']} | "
            f"{row['mean_abs_diff']} | "
            f"{row['mean_cosine_similarity']} | "
            f"{row['allclose']} | "
            f"{row['status']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If status is `passed`, the CROMA embeddings are ready for frozen-probe experiments.")
    lines.append("- The strictest alignment checks are patch order, labels, city, and region consistency across all modalities.")
    lines.append("- The pairwise checks make sure SNAP and RTC embeddings were not accidentally duplicated.")
    lines.append("- A high cosine similarity is not automatically bad; exact or near-exact equality would be suspicious.")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("After this validation passes, proceed to frozen-probe training:")
    lines.append("")
    lines.append("```text")
    lines.append("src/croma_probing/06_train_frozen_probe_224.py")
    lines.append("```")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def add_check(
    checks: List[Dict[str, object]],
    *,
    check_name: str,
    severity: str,
    passed: bool,
    details: str = "",
) -> None:
    checks.append(
        {
            "check_name": check_name,
            "severity": severity,
            "status": "passed" if passed else "failed",
            "details": details,
        }
    )


def default_modalities() -> List[str]:
    return [
        "s2",
        "s1_snap_vv_vh",
        "s1_rtc_vv_vh",
        "s2_s1_snap_vv_vh",
        "s2_s1_rtc_vv_vh",
    ]


def expected_embedding_key(modality: str) -> str:
    if modality == "s2":
        return "optical_GAP"

    if modality in {"s1_snap_vv_vh", "s1_rtc_vv_vh"}:
        return "SAR_GAP"

    if modality in {"s2_s1_snap_vv_vh", "s2_s1_rtc_vv_vh"}:
        return "joint_GAP"

    return ""


def embedding_path_for_modality(
    embedding_dir: Path,
    modality: str,
    stem: str,
) -> Path:
    return embedding_dir / f"croma_embeddings_{modality}_{stem}.npz"


def load_npz_arrays(path: Path, allow_pickle_fallback: bool) -> Dict[str, np.ndarray]:
    if not path.exists():
        fail(f"Embedding file does not exist: {path_to_str(path)}")

    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    except ValueError as exc:
        if not allow_pickle_fallback:
            raise

        log(
            "WARN",
            f"Loading with allow_pickle=True because allow_pickle=False failed for {path_to_str(path)}: {repr(exc)}",
        )

        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}


def required_npz_keys() -> List[str]:
    return [
        "embeddings",
        "manifest_row_ids",
        "patch_ids",
        "cities",
        "regions",
        "label_binary",
        "label_positive_pixels",
        "label_positive_percent",
        "label_density_bins",
        "modality",
        "croma_model_modality",
        "embedding_key",
        "normalization",
        "model_size",
        "image_resolution",
    ]


def array_to_str_list(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(str)


def validate_one_embedding_file(
    *,
    modality: str,
    path: Path,
    arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
    checks: List[Dict[str, object]],
) -> Dict[str, object]:
    missing_keys = [key for key in required_npz_keys() if key not in arrays]

    add_check(
        checks,
        check_name=f"{modality}_required_keys",
        severity="error",
        passed=len(missing_keys) == 0,
        details="" if not missing_keys else "Missing: " + "; ".join(missing_keys),
    )

    if missing_keys:
        return {
            "modality": modality,
            "status": "failed",
            "path": path_to_str(path),
            "n_rows": "",
            "embedding_dim": "",
            "embedding_key": "",
            "positive_count": "",
            "empty_count": "",
            "all_finite": "",
            "global_mean": "",
            "global_std": "",
            "mean_l2_norm": "",
            "min_l2_norm": "",
            "max_l2_norm": "",
            "zero_variance_dimensions": "",
            "file_size_mb": round_float(path.stat().st_size / (1024 ** 2), 3) if path.exists() else "",
            "notes": "Missing required keys.",
        }

    embeddings = arrays["embeddings"]
    patch_ids = array_to_str_list(arrays["patch_ids"])
    manifest_row_ids = array_to_str_list(arrays["manifest_row_ids"])
    label_binary = np.asarray(arrays["label_binary"]).astype(np.int64)

    expected_rows = int(args.expected_rows)
    expected_dim = int(args.expected_embedding_dim)

    shape_ok = embeddings.ndim == 2 and embeddings.shape == (expected_rows, expected_dim)

    add_check(
        checks,
        check_name=f"{modality}_embedding_shape",
        severity="error",
        passed=shape_ok,
        details=f"observed={tuple(embeddings.shape)}, expected=({expected_rows}, {expected_dim})",
    )

    metadata_length_ok = (
        len(patch_ids) == expected_rows
        and len(manifest_row_ids) == expected_rows
        and len(label_binary) == expected_rows
        and len(arrays["cities"]) == expected_rows
        and len(arrays["regions"]) == expected_rows
        and len(arrays["label_positive_percent"]) == expected_rows
    )

    add_check(
        checks,
        check_name=f"{modality}_metadata_lengths",
        severity="error",
        passed=metadata_length_ok,
        details=f"patch_ids={len(patch_ids)}, labels={len(label_binary)}, expected={expected_rows}",
    )

    duplicate_patch_ids = [
        patch_id for patch_id, count in Counter(patch_ids.tolist()).items()
        if count > 1
    ]

    add_check(
        checks,
        check_name=f"{modality}_unique_patch_ids",
        severity="error",
        passed=len(duplicate_patch_ids) == 0,
        details="" if not duplicate_patch_ids else f"duplicates={len(duplicate_patch_ids)}",
    )

    all_finite = bool(np.isfinite(embeddings).all())

    add_check(
        checks,
        check_name=f"{modality}_embeddings_finite",
        severity="error",
        passed=all_finite,
        details="No NaN/Inf detected." if all_finite else "NaN or Inf detected.",
    )

    global_mean = float(np.mean(embeddings)) if embeddings.size else 0.0
    global_std = float(np.std(embeddings)) if embeddings.size else 0.0

    variance_ok = global_std > float(args.min_global_std)

    add_check(
        checks,
        check_name=f"{modality}_nonzero_global_variance",
        severity="error",
        passed=variance_ok,
        details=f"global_std={global_std}, min_global_std={args.min_global_std}",
    )

    per_dim_std = np.std(embeddings, axis=0)
    zero_variance_dimensions = int(np.count_nonzero(per_dim_std <= float(args.zero_variance_tolerance)))

    zero_dim_ok = zero_variance_dimensions < expected_dim

    add_check(
        checks,
        check_name=f"{modality}_not_all_dimensions_zero_variance",
        severity="error",
        passed=zero_dim_ok,
        details=f"zero_variance_dimensions={zero_variance_dimensions}/{expected_dim}",
    )

    l2_norms = np.linalg.norm(embeddings, axis=1)
    nonzero_norms = bool(np.all(l2_norms > float(args.min_l2_norm)))

    add_check(
        checks,
        check_name=f"{modality}_nonzero_embedding_norms",
        severity="error",
        passed=nonzero_norms,
        details=f"min_l2_norm={float(np.min(l2_norms)) if l2_norms.size else 0.0}",
    )

    positive_count = int(np.count_nonzero(label_binary == 1))
    empty_count = int(np.count_nonzero(label_binary == 0))

    label_counts_ok = (
        positive_count == int(args.expected_positive)
        and empty_count == int(args.expected_empty)
    )

    add_check(
        checks,
        check_name=f"{modality}_label_counts",
        severity="error",
        passed=label_counts_ok,
        details=f"positive={positive_count}, empty={empty_count}",
    )

    modality_value = normalize_scalar_array_value(arrays["modality"])
    modality_ok = modality_value == modality

    add_check(
        checks,
        check_name=f"{modality}_stored_modality_name",
        severity="error",
        passed=modality_ok,
        details=f"stored={modality_value}, expected={modality}",
    )

    stored_embedding_key = normalize_scalar_array_value(arrays["embedding_key"])
    expected_key = expected_embedding_key(modality)

    embedding_key_ok = stored_embedding_key == expected_key

    add_check(
        checks,
        check_name=f"{modality}_embedding_key",
        severity="error",
        passed=embedding_key_ok,
        details=f"stored={stored_embedding_key}, expected={expected_key}",
    )

    image_resolution_value = safe_int(normalize_scalar_array_value(arrays["image_resolution"]))
    image_resolution_ok = image_resolution_value == int(args.image_resolution)

    add_check(
        checks,
        check_name=f"{modality}_image_resolution",
        severity="error",
        passed=image_resolution_ok,
        details=f"stored={image_resolution_value}, expected={args.image_resolution}",
    )

    status = "passed" if (
        shape_ok
        and metadata_length_ok
        and len(duplicate_patch_ids) == 0
        and all_finite
        and variance_ok
        and zero_dim_ok
        and nonzero_norms
        and label_counts_ok
        and modality_ok
        and embedding_key_ok
        and image_resolution_ok
    ) else "failed"

    return {
        "modality": modality,
        "status": status,
        "path": path_to_str(path),
        "n_rows": int(embeddings.shape[0]) if embeddings.ndim >= 1 else "",
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else "",
        "embedding_key": stored_embedding_key,
        "positive_count": positive_count,
        "empty_count": empty_count,
        "all_finite": all_finite,
        "global_mean": round_float(global_mean, 8),
        "global_std": round_float(global_std, 8),
        "mean_l2_norm": round_float(float(np.mean(l2_norms)), 8) if l2_norms.size else 0.0,
        "min_l2_norm": round_float(float(np.min(l2_norms)), 8) if l2_norms.size else 0.0,
        "max_l2_norm": round_float(float(np.max(l2_norms)), 8) if l2_norms.size else 0.0,
        "zero_variance_dimensions": zero_variance_dimensions,
        "file_size_mb": round_float(path.stat().st_size / (1024 ** 2), 3),
        "notes": "",
    }


def compare_metadata_alignment(
    *,
    reference_modality: str,
    reference_arrays: Dict[str, np.ndarray],
    modality: str,
    arrays: Dict[str, np.ndarray],
    checks: List[Dict[str, object]],
) -> None:
    comparison_keys = [
        "patch_ids",
        "label_binary",
        "cities",
        "regions",
        "label_positive_pixels",
        "label_density_bins",
    ]

    for key in comparison_keys:
        ref = reference_arrays[key]
        cur = arrays[key]

        same = bool(np.array_equal(ref, cur))

        add_check(
            checks,
            check_name=f"{modality}_aligned_{key}_with_{reference_modality}",
            severity="error",
            passed=same,
            details=f"Compared {key} with reference modality {reference_modality}.",
        )

    ref_percent = np.asarray(reference_arrays["label_positive_percent"]).astype(np.float32)
    cur_percent = np.asarray(arrays["label_positive_percent"]).astype(np.float32)

    same_percent = bool(np.allclose(ref_percent, cur_percent, atol=1e-6, rtol=0.0))

    add_check(
        checks,
        check_name=f"{modality}_aligned_label_positive_percent_with_{reference_modality}",
        severity="error",
        passed=same_percent,
        details=f"Compared label_positive_percent with reference modality {reference_modality}.",
    )


def pairwise_embedding_checks(
    *,
    arrays_by_modality: Dict[str, Dict[str, np.ndarray]],
    args: argparse.Namespace,
    checks: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    pairwise_rows: List[Dict[str, object]] = []

    for modality_a, modality_b in combinations(args.modalities, 2):
        emb_a = arrays_by_modality[modality_a]["embeddings"].astype(np.float32)
        emb_b = arrays_by_modality[modality_b]["embeddings"].astype(np.float32)

        same_shape = emb_a.shape == emb_b.shape

        if not same_shape:
            row = {
                "modality_a": modality_a,
                "modality_b": modality_b,
                "same_shape": False,
                "max_abs_diff": "",
                "mean_abs_diff": "",
                "mean_cosine_similarity": "",
                "allclose": "",
                "status": "failed",
                "notes": "Embedding shapes differ.",
            }

            pairwise_rows.append(row)

            add_check(
                checks,
                check_name=f"pairwise_{modality_a}_vs_{modality_b}_same_shape",
                severity="error",
                passed=False,
                details=f"{emb_a.shape} vs {emb_b.shape}",
            )

            continue

        diff = np.abs(emb_a - emb_b)
        max_abs_diff = float(np.max(diff))
        mean_abs_diff = float(np.mean(diff))

        denom = (
            np.linalg.norm(emb_a, axis=1)
            * np.linalg.norm(emb_b, axis=1)
            + 1e-12
        )

        cosine = np.sum(emb_a * emb_b, axis=1) / denom
        mean_cosine = float(np.mean(cosine))

        allclose = bool(np.allclose(
            emb_a,
            emb_b,
            atol=float(args.identical_atol),
            rtol=float(args.identical_rtol),
        ))

        status = "failed" if allclose else "passed"

        pairwise_rows.append(
            {
                "modality_a": modality_a,
                "modality_b": modality_b,
                "same_shape": True,
                "max_abs_diff": round_float(max_abs_diff, 10),
                "mean_abs_diff": round_float(mean_abs_diff, 10),
                "mean_cosine_similarity": round_float(mean_cosine, 10),
                "allclose": allclose,
                "status": status,
                "notes": "" if not allclose else "Embeddings are suspiciously identical.",
            }
        )

        add_check(
            checks,
            check_name=f"pairwise_{modality_a}_vs_{modality_b}_not_identical",
            severity="error",
            passed=not allclose,
            details=(
                f"max_abs_diff={max_abs_diff}, "
                f"mean_abs_diff={mean_abs_diff}, "
                f"mean_cosine={mean_cosine}"
            ),
        )

    return pairwise_rows


def build_summary(
    *,
    instance_root: Path,
    embedding_dir: Path,
    modality_rows: List[Dict[str, object]],
    check_rows: List[Dict[str, object]],
    pairwise_rows: List[Dict[str, object]],
    args: argparse.Namespace,
    output_paths: Dict[str, Path],
) -> Dict[str, object]:
    error_count = sum(
        1 for row in check_rows
        if row["severity"] == "error" and row["status"] != "passed"
    )

    warning_count = sum(
        1 for row in check_rows
        if row["severity"] == "warning" and row["status"] != "passed"
    )

    return {
        "created_utc": now_utc(),
        "status": "passed" if error_count == 0 else "failed",
        "error_count": error_count,
        "warning_count": warning_count,
        "instance_root": path_to_str(instance_root),
        "embedding_dir": path_to_str(embedding_dir),
        "modalities": list(args.modalities),
        "expected_rows": int(args.expected_rows),
        "expected_embedding_dim": int(args.expected_embedding_dim),
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "expected_positive": args.expected_positive,
            "expected_empty": args.expected_empty,
            "image_resolution": args.image_resolution,
            "min_global_std": args.min_global_std,
            "min_l2_norm": args.min_l2_norm,
            "zero_variance_tolerance": args.zero_variance_tolerance,
            "identical_atol": args.identical_atol,
            "identical_rtol": args.identical_rtol,
        },
        "outputs": {key: path_to_str(value) for key, value in output_paths.items()},
        "modality_rows": modality_rows,
        "pairwise_rows": pairwise_rows,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate full CROMA embedding files before frozen-probe training."
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
        help="Default: same as embedding-dir.",
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
        help="Modalities to validate.",
    )

    parser.add_argument(
        "--expected-rows",
        type=int,
        default=12699,
        help="Expected number of patch embeddings per modality. Default: 12699.",
    )

    parser.add_argument(
        "--expected-embedding-dim",
        type=int,
        default=768,
        help="Expected CROMA base GAP embedding dimension. Default: 768.",
    )

    parser.add_argument(
        "--expected-positive",
        type=int,
        default=6382,
        help="Expected positive patch count. Default: 6382.",
    )

    parser.add_argument(
        "--expected-empty",
        type=int,
        default=6317,
        help="Expected empty patch count. Default: 6317.",
    )

    parser.add_argument(
        "--image-resolution",
        type=int,
        default=224,
        help="Expected stored image resolution. Default: 224.",
    )

    parser.add_argument(
        "--reference-modality",
        default="s2",
        help="Reference modality for metadata alignment checks. Default: s2.",
    )

    parser.add_argument(
        "--min-global-std",
        type=float,
        default=1e-8,
        help="Minimum allowed global embedding standard deviation. Default: 1e-8.",
    )

    parser.add_argument(
        "--min-l2-norm",
        type=float,
        default=1e-8,
        help="Minimum allowed per-row L2 norm. Default: 1e-8.",
    )

    parser.add_argument(
        "--zero-variance-tolerance",
        type=float,
        default=1e-12,
        help="Tolerance for counting zero-variance dimensions. Default: 1e-12.",
    )

    parser.add_argument(
        "--identical-atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for pairwise allclose duplicate check. Default: 1e-8.",
    )

    parser.add_argument(
        "--identical-rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for pairwise allclose duplicate check. Default: 1e-6.",
    )

    parser.add_argument(
        "--allow-pickle-fallback",
        action="store_true",
        help="Allow loading legacy object arrays with allow_pickle=True if needed.",
    )

    parser.add_argument(
        "--no-fail-on-error",
        action="store_true",
        help="Write reports but do not exit with non-zero code on validation failure.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite validation outputs.",
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

    output_dir: Path = args.output_dir or embedding_dir

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    summary_csv = output_dir / f"croma_embedding_validation_summary_{stem}.csv"
    checks_csv = output_dir / f"croma_embedding_validation_checks_{stem}.csv"
    pairwise_csv = output_dir / f"croma_embedding_validation_pairwise_{stem}.csv"
    json_path = output_dir / f"croma_embedding_validation_{stem}.json"
    md_path = output_dir / f"croma_embedding_validation_{stem}.md"

    output_paths = {
        "summary_csv": summary_csv,
        "checks_csv": checks_csv,
        "pairwise_csv": pairwise_csv,
        "json": json_path,
        "markdown": md_path,
    }

    log("STEP", "Validating full CROMA embeddings.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Embedding dir: {path_to_str(embedding_dir)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")
    log("INFO", f"Modalities:    {';'.join(args.modalities)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    if not embedding_dir.exists():
        fail(f"Embedding dir does not exist: {path_to_str(embedding_dir)}")

    if args.reference_modality not in args.modalities:
        fail(
            f"Reference modality `{args.reference_modality}` is not in modalities: "
            f"{args.modalities}"
        )

    checks: List[Dict[str, object]] = []
    modality_rows: List[Dict[str, object]] = []
    arrays_by_modality: Dict[str, Dict[str, np.ndarray]] = {}

    for modality in args.modalities:
        path = embedding_path_for_modality(embedding_dir, modality, stem)

        add_check(
            checks,
            check_name=f"{modality}_file_exists",
            severity="error",
            passed=path.exists(),
            details=path_to_str(path),
        )

        if not path.exists():
            modality_rows.append(
                {
                    "modality": modality,
                    "status": "failed",
                    "path": path_to_str(path),
                    "n_rows": "",
                    "embedding_dim": "",
                    "embedding_key": "",
                    "positive_count": "",
                    "empty_count": "",
                    "all_finite": "",
                    "global_mean": "",
                    "global_std": "",
                    "mean_l2_norm": "",
                    "min_l2_norm": "",
                    "max_l2_norm": "",
                    "zero_variance_dimensions": "",
                    "file_size_mb": "",
                    "notes": "File does not exist.",
                }
            )
            continue

        arrays = load_npz_arrays(path, allow_pickle_fallback=bool(args.allow_pickle_fallback))
        arrays_by_modality[modality] = arrays

        row = validate_one_embedding_file(
            modality=modality,
            path=path,
            arrays=arrays,
            args=args,
            checks=checks,
        )

        modality_rows.append(row)

        log(
            "OK" if row["status"] == "passed" else "ERROR",
            f"{modality}: status={row['status']}, "
            f"shape=({row['n_rows']}, {row['embedding_dim']}), "
            f"std={row['global_std']}",
        )

    if args.reference_modality in arrays_by_modality:
        ref_arrays = arrays_by_modality[args.reference_modality]

        for modality, arrays in arrays_by_modality.items():
            if modality == args.reference_modality:
                continue

            compare_metadata_alignment(
                reference_modality=args.reference_modality,
                reference_arrays=ref_arrays,
                modality=modality,
                arrays=arrays,
                checks=checks,
            )

    pairwise_rows = []

    if len(arrays_by_modality) >= 2:
        pairwise_rows = pairwise_embedding_checks(
            arrays_by_modality=arrays_by_modality,
            args=args,
            checks=checks,
        )

    summary = build_summary(
        instance_root=instance_root,
        embedding_dir=embedding_dir,
        modality_rows=modality_rows,
        check_rows=checks,
        pairwise_rows=pairwise_rows,
        args=args,
        output_paths=output_paths,
    )

    log("STEP", "Writing embedding validation outputs.")

    write_csv(
        summary_csv,
        modality_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "modality",
            "status",
            "path",
            "n_rows",
            "embedding_dim",
            "embedding_key",
            "positive_count",
            "empty_count",
            "all_finite",
            "global_mean",
            "global_std",
            "mean_l2_norm",
            "min_l2_norm",
            "max_l2_norm",
            "zero_variance_dimensions",
            "file_size_mb",
            "notes",
        ],
    )

    write_csv(
        checks_csv,
        checks,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "check_name",
            "severity",
            "status",
            "details",
        ],
    )

    write_csv(
        pairwise_csv,
        pairwise_rows,
        overwrite=bool(args.overwrite),
        fieldnames=[
            "modality_a",
            "modality_b",
            "same_shape",
            "max_abs_diff",
            "mean_abs_diff",
            "mean_cosine_similarity",
            "allclose",
            "status",
            "notes",
        ],
    )

    write_json(json_path, summary, overwrite=bool(args.overwrite))
    write_markdown(md_path, summary, modality_rows, checks, pairwise_rows, overwrite=bool(args.overwrite))

    log("OK", f"Wrote summary CSV:  {path_to_str(summary_csv)}")
    log("OK", f"Wrote checks CSV:   {path_to_str(checks_csv)}")
    log("OK", f"Wrote pairwise CSV: {path_to_str(pairwise_csv)}")
    log("OK", f"Wrote JSON:         {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:     {path_to_str(md_path)}")

    log("STEP", "Final embedding validation summary.")
    log("OK" if summary["status"] == "passed" else "ERROR", f"Status: {summary['status']}")
    log("OK", f"Error count: {summary['error_count']}")
    log("OK", f"Warning count: {summary['warning_count']}")

    if summary["status"] != "passed" and not args.no_fail_on_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()