"""
2_inspect_croma_embedding_shapes_224.py

Main objective
--------------
Inspect existing CROMA embedding .npz files and determine whether they contain
only global patch embeddings or dense spatial features usable for UPerNet.

Why this matters
----------------
UPerNet is a dense segmentation decoder. It cannot train properly from only
global patch vectors shaped like:

    N x 768

For segmentation, we need dense spatial features/tokens such as:

    N x 768 x 14 x 14

or token sequences such as:

    N x 196 x 768

for 224 x 224 input patches with 16 x 16 ViT-like tokens.

This script does NOT train anything.
It only inspects files and writes a clear diagnostic report.

Recommended command
-------------------
python src/splitting_strategy_experiments/2_inspect_croma_embedding_shapes_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --target-modality "s2_s1_snap_vv_vh" `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib import format as npy_format


EXPECTED_PATCH_COUNT_DEFAULT = 12699
EXPECTED_FEATURE_DIM_DEFAULT = 768


def log(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"[WARNING] {message}", flush=True)


def fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def default_embeddings_dir(instance_root: Path) -> Path:
    return instance_root / "metadata" / "croma_probing" / "full_embeddings"


def default_manifest_path(instance_root: Path, patch_size: int, stride: int, edge_mode: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / f"croma_comparison_manifest_ps{patch_size}_st{stride}_{edge_mode}.csv"
    )


def default_output_dir(instance_root: Path, patch_size: int, stride: int, edge_mode: str) -> Path:
    return (
        instance_root
        / "metadata"
        / "splitting_strategy_experiments"
        / f"croma_embedding_shape_inspection_ps{patch_size}_st{stride}_{edge_mode}"
    )


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = (
        list(output_dir.glob("*.csv"))
        + list(output_dir.glob("*.json"))
        + list(output_dir.glob("*.md"))
    )

    if existing and not overwrite:
        fail(
            f"Output directory already contains files:\n{output_dir}\n\n"
            f"Use --overwrite to replace them."
        )


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [jsonable(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if math.isnan(v) else v

    if isinstance(value, float):
        return None if math.isnan(value) else value

    if isinstance(value, Path):
        return str(value)

    return value


def infer_expected_patch_count_from_manifest(
    manifest_path: Path,
    target_modality: str,
) -> Optional[int]:
    if not manifest_path.exists():
        warn(f"Manifest not found, cannot infer expected patch count:\n{manifest_path}")
        return None

    try:
        df = pd.read_csv(manifest_path)
    except Exception as exc:
        warn(f"Could not read manifest to infer expected patch count: {exc}")
        return None

    if df.empty:
        warn("Manifest is empty; cannot infer expected patch count.")
        return None

    if "modality" in df.columns:
        target_norm = normalize_text(target_modality)
        modality_norm = df["modality"].map(normalize_text)
        df = df[modality_norm == target_norm].copy()

        if df.empty:
            warn(
                f"Manifest contains a modality column, but no rows matched target modality "
                f"'{target_modality}'. Expected patch count will be inferred from all rows if possible."
            )
            df = pd.read_csv(manifest_path)

    if "patch_id" in df.columns:
        return int(df["patch_id"].astype(str).nunique())

    return int(len(df))


def read_npy_header_from_npz_member(
    zf: zipfile.ZipFile,
    member_name: str,
) -> Tuple[Tuple[int, ...], str, bool]:
    """
    Read shape, dtype, and fortran_order from a .npy member inside a .npz file
    without loading the full array into memory.

    This is important because dense CROMA feature files could be very large.
    """
    with zf.open(member_name, "r") as f:
        version = npy_format.read_magic(f)

        if version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(f)
        elif version == (3, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported .npy version {version} in member {member_name}")

    return tuple(int(x) for x in shape), str(dtype), bool(fortran_order)


def npz_key_from_member_name(member_name: str) -> str:
    if member_name.endswith(".npy"):
        return member_name[:-4]
    return member_name


def inspect_npz_headers(npz_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    try:
        with zipfile.ZipFile(npz_path, "r") as zf:
            members = [name for name in zf.namelist() if name.endswith(".npy")]

            if not members:
                warn(f"No .npy arrays found inside {npz_path.name}")
                return rows

            for member in members:
                try:
                    shape, dtype, fortran_order = read_npy_header_from_npz_member(zf, member)
                    info = zf.getinfo(member)

                    rows.append(
                        {
                            "key": npz_key_from_member_name(member),
                            "shape": shape,
                            "ndim": len(shape),
                            "dtype": dtype,
                            "fortran_order": fortran_order,
                            "compressed_size_bytes": int(info.compress_size),
                            "uncompressed_size_bytes": int(info.file_size),
                        }
                    )

                except Exception as exc:
                    rows.append(
                        {
                            "key": npz_key_from_member_name(member),
                            "shape": None,
                            "ndim": None,
                            "dtype": None,
                            "fortran_order": None,
                            "compressed_size_bytes": None,
                            "uncompressed_size_bytes": None,
                            "error": str(exc),
                        }
                    )

    except zipfile.BadZipFile:
        fail(f"File is not a valid .npz/.zip file: {npz_path}")

    return rows


def infer_modality_from_filename(filename: str) -> str:
    stem = Path(filename).stem

    # Expected pattern:
    # croma_embeddings_s2_s1_snap_vv_vh_ps224_st112_cover
    m = re.match(r"^croma_embeddings_(.+?)_ps\d+_st\d+_.+$", stem)
    if m:
        return m.group(1)

    # Fallback
    stem = stem.replace("croma_embeddings_", "")
    stem = re.sub(r"_ps\d+_st\d+_.*$", "", stem)
    return stem


def is_probably_metadata_key(key: str, dtype: Optional[str], shape: Optional[Tuple[int, ...]]) -> bool:
    key_norm = normalize_text(key)

    metadata_tokens = [
        "patch_id",
        "city",
        "region",
        "label",
        "target",
        "positive",
        "path",
        "row",
        "col",
        "x",
        "y",
        "window",
        "split",
        "modality",
        "index",
        "filename",
        "file",
        "id",
    ]

    if any(tok in key_norm for tok in metadata_tokens):
        return True

    if dtype is not None:
        dtype_norm = str(dtype).lower()
        if "str" in dtype_norm or "unicode" in dtype_norm or "object" in dtype_norm:
            return True

    if shape is not None and len(shape) == 1:
        return True

    return False


def classify_array_shape(
    key: str,
    shape: Optional[Tuple[int, ...]],
    dtype: Optional[str],
    expected_n: Optional[int],
    expected_c: int,
) -> Dict[str, Any]:
    """
    Classify an array as global embedding, dense token sequence, dense feature map,
    metadata, or unknown.
    """
    result: Dict[str, Any] = {
        "array_role": "unknown",
        "is_candidate_feature": False,
        "is_global_embedding": False,
        "is_dense_tokens": False,
        "is_dense_feature_map": False,
        "is_metadata": False,
        "sample_count_matches_expected": None,
        "feature_dim_matches_expected": None,
        "inferred_hw": None,
        "upernet_usable": False,
        "reason": "",
    }

    if shape is None:
        result["reason"] = "shape could not be read"
        return result

    ndim = len(shape)

    if ndim == 0:
        result["array_role"] = "scalar_or_empty"
        result["reason"] = "0D array"
        return result

    n0 = shape[0]
    if expected_n is not None:
        result["sample_count_matches_expected"] = bool(n0 == expected_n)

    if is_probably_metadata_key(key, dtype, shape):
        result["array_role"] = "metadata_or_label"
        result["is_metadata"] = True
        result["reason"] = "key/dtype/shape looks like metadata or labels"
        return result

    # 2D: usually global embeddings N x C
    if ndim == 2:
        n, c = shape
        result["feature_dim_matches_expected"] = bool(c == expected_c)

        if expected_n is not None and n == expected_n and c == expected_c:
            result["array_role"] = "global_patch_embedding"
            result["is_candidate_feature"] = True
            result["is_global_embedding"] = True
            result["upernet_usable"] = False
            result["reason"] = (
                "N x C global patch embedding. Useful for frozen-probe classification, "
                "but not directly sufficient for UPerNet dense segmentation."
            )
            return result

        result["array_role"] = "two_dimensional_array"
        result["reason"] = "2D array but not a clean N x expected_C global embedding"
        return result

    # 3D: could be token sequence N x tokens x C or N x C x tokens
    if ndim == 3:
        n, a, b = shape

        if expected_n is not None and n == expected_n:
            if b == expected_c:
                tokens = a
                hw = int(round(math.sqrt(tokens)))
                result["feature_dim_matches_expected"] = True
                result["array_role"] = "dense_token_sequence_n_tokens_c"
                result["is_candidate_feature"] = True
                result["is_dense_tokens"] = True
                result["inferred_hw"] = f"{hw}x{hw}" if hw * hw == tokens else None
                result["upernet_usable"] = True
                result["reason"] = (
                    "N x tokens x C dense token sequence. This can probably be reshaped "
                    "for UPerNet if tokens form a square grid."
                )
                return result

            if a == expected_c:
                tokens = b
                hw = int(round(math.sqrt(tokens)))
                result["feature_dim_matches_expected"] = True
                result["array_role"] = "dense_token_sequence_n_c_tokens"
                result["is_candidate_feature"] = True
                result["is_dense_tokens"] = True
                result["inferred_hw"] = f"{hw}x{hw}" if hw * hw == tokens else None
                result["upernet_usable"] = True
                result["reason"] = (
                    "N x C x tokens dense token sequence. This can probably be reshaped "
                    "for UPerNet if tokens form a square grid."
                )
                return result

        result["array_role"] = "three_dimensional_array"
        result["reason"] = "3D array, but shape does not clearly match dense token features"
        return result

    # 4D: likely dense feature map N x C x H x W or N x H x W x C
    if ndim == 4:
        n, d1, d2, d3 = shape

        if expected_n is not None and n == expected_n:
            if d1 == expected_c:
                result["feature_dim_matches_expected"] = True
                result["array_role"] = "dense_feature_map_n_c_h_w"
                result["is_candidate_feature"] = True
                result["is_dense_feature_map"] = True
                result["inferred_hw"] = f"{d2}x{d3}"
                result["upernet_usable"] = True
                result["reason"] = "N x C x H x W dense feature map. Directly suitable for UPerNet-style decoder."
                return result

            if d3 == expected_c:
                result["feature_dim_matches_expected"] = True
                result["array_role"] = "dense_feature_map_n_h_w_c"
                result["is_candidate_feature"] = True
                result["is_dense_feature_map"] = True
                result["inferred_hw"] = f"{d1}x{d2}"
                result["upernet_usable"] = True
                result["reason"] = (
                    "N x H x W x C dense feature map. Suitable after transposing to N x C x H x W."
                )
                return result

        result["array_role"] = "four_dimensional_array"
        result["reason"] = "4D array, but shape does not clearly match dense feature maps"
        return result

    if ndim == 5:
        result["array_role"] = "possible_multilayer_dense_features"
        result["is_candidate_feature"] = True
        result["reason"] = (
            "5D array. This may represent multiple layers or views. It needs manual inspection."
        )
        return result

    result["array_role"] = f"{ndim}d_array"
    result["reason"] = f"{ndim}D array; not classified automatically"
    return result


def inspect_embeddings(
    embeddings_dir: Path,
    expected_n: Optional[int],
    expected_c: int,
    target_modality: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not embeddings_dir.exists():
        fail(f"Embeddings directory does not exist:\n{embeddings_dir}")

    npz_files = sorted(embeddings_dir.glob("*.npz"))

    if not npz_files:
        fail(f"No .npz files found in:\n{embeddings_dir}")

    all_rows: List[Dict[str, Any]] = []
    file_summaries: Dict[str, Any] = {}

    log(f"[2/6] Found {len(npz_files):,} .npz files")

    for npz_path in npz_files:
        log(f"      Inspecting: {npz_path.name}")

        modality = infer_modality_from_filename(npz_path.name)
        modality_norm = normalize_text(modality)
        target_norm = normalize_text(target_modality)

        header_rows = inspect_npz_headers(npz_path)

        file_has_global = False
        file_has_dense = False
        file_has_upernet_usable = False
        file_errors: List[str] = []

        for row in header_rows:
            shape = row.get("shape")
            dtype = row.get("dtype")
            key = row.get("key")

            if "error" in row:
                file_errors.append(f"{key}: {row['error']}")

            classification = classify_array_shape(
                key=str(key),
                shape=shape,
                dtype=dtype,
                expected_n=expected_n,
                expected_c=expected_c,
            )

            file_has_global = file_has_global or bool(classification["is_global_embedding"])
            file_has_dense = file_has_dense or bool(
                classification["is_dense_tokens"] or classification["is_dense_feature_map"]
            )
            file_has_upernet_usable = file_has_upernet_usable or bool(classification["upernet_usable"])

            flat_row: Dict[str, Any] = {
                "file": npz_path.name,
                "file_path": str(npz_path),
                "modality_from_filename": modality,
                "is_target_modality": bool(modality_norm == target_norm),
                "key": row.get("key"),
                "shape": str(row.get("shape")),
                "shape_json": json.dumps(row.get("shape")),
                "ndim": row.get("ndim"),
                "dtype": row.get("dtype"),
                "fortran_order": row.get("fortran_order"),
                "compressed_size_mb": (
                    float(row["compressed_size_bytes"]) / (1024.0 ** 2)
                    if row.get("compressed_size_bytes") is not None
                    else None
                ),
                "uncompressed_size_mb": (
                    float(row["uncompressed_size_bytes"]) / (1024.0 ** 2)
                    if row.get("uncompressed_size_bytes") is not None
                    else None
                ),
                **classification,
            }

            all_rows.append(flat_row)

        file_summaries[npz_path.name] = {
            "file_path": str(npz_path),
            "modality_from_filename": modality,
            "is_target_modality": bool(modality_norm == target_norm),
            "n_arrays": len(header_rows),
            "has_global_embedding": file_has_global,
            "has_dense_features": file_has_dense,
            "has_upernet_usable_features": file_has_upernet_usable,
            "errors": file_errors,
        }

    details_df = pd.DataFrame(all_rows)

    return details_df, file_summaries


def summarize_decision(
    details_df: pd.DataFrame,
    file_summaries: Dict[str, Any],
    target_modality: str,
    expected_n: Optional[int],
) -> Dict[str, Any]:
    target_rows = details_df[details_df["is_target_modality"] == True].copy()

    target_files = [
        file_name
        for file_name, summary in file_summaries.items()
        if summary.get("is_target_modality", False)
    ]

    has_target_file = len(target_files) > 0

    target_has_global = bool(target_rows["is_global_embedding"].fillna(False).any()) if not target_rows.empty else False
    target_has_dense_tokens = bool(target_rows["is_dense_tokens"].fillna(False).any()) if not target_rows.empty else False
    target_has_dense_map = bool(target_rows["is_dense_feature_map"].fillna(False).any()) if not target_rows.empty else False
    target_has_upernet_usable = bool(target_rows["upernet_usable"].fillna(False).any()) if not target_rows.empty else False

    all_has_upernet_usable = bool(details_df["upernet_usable"].fillna(False).any()) if not details_df.empty else False

    if not has_target_file:
        recommendation = "target_modality_file_missing"
        conclusion = (
            f"No embedding file matching target modality '{target_modality}' was found. "
            f"Check the embedding filename or rerun CROMA extraction."
        )
        can_proceed_to_upernet = False

    elif target_has_upernet_usable:
        recommendation = "dense_features_available"
        conclusion = (
            f"The target modality '{target_modality}' appears to contain dense spatial features. "
            f"We can proceed to dense feature validation and then UPerNet dataset/model implementation."
        )
        can_proceed_to_upernet = True

    elif target_has_global and not target_has_upernet_usable:
        recommendation = "global_only_extract_dense_features_next"
        conclusion = (
            f"The target modality '{target_modality}' appears to contain only global patch embeddings. "
            f"These are not sufficient for UPerNet segmentation. "
            f"The next step should be dense CROMA feature extraction."
        )
        can_proceed_to_upernet = False

    else:
        recommendation = "manual_inspection_needed"
        conclusion = (
            f"The target modality '{target_modality}' exists, but no clearly usable dense features "
            f"or standard global embedding were identified. Manual inspection is needed."
        )
        can_proceed_to_upernet = False

    return {
        "target_modality": target_modality,
        "expected_patch_count": expected_n,
        "target_files": target_files,
        "has_target_file": has_target_file,
        "target_has_global_embedding": target_has_global,
        "target_has_dense_tokens": target_has_dense_tokens,
        "target_has_dense_feature_map": target_has_dense_map,
        "target_has_upernet_usable_features": target_has_upernet_usable,
        "any_file_has_upernet_usable_features": all_has_upernet_usable,
        "recommendation": recommendation,
        "can_proceed_directly_to_upernet": can_proceed_to_upernet,
        "conclusion": conclusion,
    }


def write_markdown_report(
    path: Path,
    decision: Dict[str, Any],
    file_summaries: Dict[str, Any],
    details_df: pd.DataFrame,
    embeddings_dir: Path,
    manifest_path: Path,
    expected_c: int,
) -> None:
    lines: List[str] = []

    lines.append("# CROMA Embedding Shape Inspection Report")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Inspect existing CROMA `.npz` embedding files and determine whether they contain "
        "dense spatial features usable for UPerNet segmentation, or only global patch embeddings."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Embeddings directory: `{embeddings_dir}`")
    lines.append(f"- Manifest path: `{manifest_path}`")
    lines.append(f"- Target modality: `{decision['target_modality']}`")
    lines.append(f"- Expected patch count: `{decision['expected_patch_count']}`")
    lines.append(f"- Expected feature dimension: `{expected_c}`")
    lines.append("")
    lines.append("## Final Decision")
    lines.append("")
    lines.append(f"- Recommendation: `{decision['recommendation']}`")
    lines.append(f"- Can proceed directly to UPerNet: `{decision['can_proceed_directly_to_upernet']}`")
    lines.append(f"- Target has global embedding: `{decision['target_has_global_embedding']}`")
    lines.append(f"- Target has dense token features: `{decision['target_has_dense_tokens']}`")
    lines.append(f"- Target has dense feature maps: `{decision['target_has_dense_feature_map']}`")
    lines.append("")
    lines.append("### Conclusion")
    lines.append("")
    lines.append(decision["conclusion"])
    lines.append("")
    lines.append("## File-Level Summary")
    lines.append("")
    lines.append(
        "| File | Modality | Target? | Arrays | Global? | Dense? | UPerNet-usable? |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|"
    )

    for file_name, summary in file_summaries.items():
        lines.append(
            f"| `{file_name}` "
            f"| `{summary['modality_from_filename']}` "
            f"| {summary['is_target_modality']} "
            f"| {summary['n_arrays']} "
            f"| {summary['has_global_embedding']} "
            f"| {summary['has_dense_features']} "
            f"| {summary['has_upernet_usable_features']} |"
        )

    lines.append("")
    lines.append("## Array-Level Details")
    lines.append("")
    lines.append(
        "| File | Key | Shape | Dtype | Role | UPerNet-usable? | Reason |"
    )
    lines.append(
        "|---|---|---|---|---|---:|---|"
    )

    for _, row in details_df.iterrows():
        reason = str(row.get("reason", "")).replace("|", "\\|")
        lines.append(
            f"| `{row['file']}` "
            f"| `{row['key']}` "
            f"| `{row['shape']}` "
            f"| `{row['dtype']}` "
            f"| `{row['array_role']}` "
            f"| {row['upernet_usable']} "
            f"| {reason} |"
        )

    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- `global_patch_embedding`: usually shaped `N x C`, for example `12699 x 768`. This is useful for frozen-probe classification but not sufficient for dense segmentation.")
    lines.append("- `dense_token_sequence_n_tokens_c`: usually shaped `N x 196 x 768`. This can likely be reshaped to `N x 768 x 14 x 14`.")
    lines.append("- `dense_feature_map_n_c_h_w`: already in feature-map format and directly suitable for a UPerNet-style decoder.")
    lines.append("- `dense_feature_map_n_h_w_c`: usable after transposing to `N x C x H x W`.")
    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    if decision["can_proceed_directly_to_upernet"]:
        lines.append(
            "Proceed to dense feature validation: verify patch ID alignment, feature/label alignment, "
            "and train/validation/test split compatibility."
        )
    else:
        lines.append(
            "Implement dense CROMA feature extraction before building the UPerNet dataset and model."
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    instance_root = Path(args.instance_root)
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else default_embeddings_dir(instance_root)
    manifest_path = Path(args.manifest) if args.manifest else default_manifest_path(
        instance_root=instance_root,
        patch_size=args.patch_size,
        stride=args.stride,
        edge_mode=args.edge_mode,
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
        instance_root=instance_root,
        patch_size=args.patch_size,
        stride=args.stride,
        edge_mode=args.edge_mode,
    )

    log("=" * 100)
    log("CROMA Embedding Shape Inspector")
    log("=" * 100)
    log(f"Instance root:    {instance_root}")
    log(f"Embeddings dir:   {embeddings_dir}")
    log(f"Manifest path:    {manifest_path}")
    log(f"Output dir:       {output_dir}")
    log(f"Target modality:  {args.target_modality}")
    log(f"Patch size:       {args.patch_size}")
    log(f"Stride:           {args.stride}")
    log(f"Edge mode:        {args.edge_mode}")
    log(f"Expected C:       {args.expected_feature_dim}")
    log("=" * 100)

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    log("[1/6] Inferring expected patch count")
    expected_n = args.expected_patch_count

    if expected_n is None:
        expected_n = infer_expected_patch_count_from_manifest(
            manifest_path=manifest_path,
            target_modality=args.target_modality,
        )

    if expected_n is None:
        expected_n = EXPECTED_PATCH_COUNT_DEFAULT
        warn(f"Falling back to default expected patch count: {expected_n:,}")

    log(f"      Expected patch count: {expected_n:,}")

    details_df, file_summaries = inspect_embeddings(
        embeddings_dir=embeddings_dir,
        expected_n=expected_n,
        expected_c=args.expected_feature_dim,
        target_modality=args.target_modality,
    )

    if details_df.empty:
        fail("No arrays were inspected. Something is wrong with the embedding files.")

    log("[3/6] Building decision summary")
    decision = summarize_decision(
        details_df=details_df,
        file_summaries=file_summaries,
        target_modality=args.target_modality,
        expected_n=expected_n,
    )

    log("")
    log("Decision:")
    log(f"  Recommendation: {decision['recommendation']}")
    log(f"  Can proceed directly to UPerNet: {decision['can_proceed_directly_to_upernet']}")
    log(f"  Target files: {decision['target_files']}")
    log(f"  Conclusion: {decision['conclusion']}")
    log("")

    log("[4/6] Writing outputs")

    details_csv = output_dir / f"croma_embedding_shape_details_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    summary_json = output_dir / f"croma_embedding_shape_summary_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.json"
    report_md = output_dir / f"croma_embedding_shape_report_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.md"

    if not args.dry_run:
        details_df.to_csv(details_csv, index=False)

        summary_payload = {
            "configuration": {
                "instance_root": str(instance_root),
                "embeddings_dir": str(embeddings_dir),
                "manifest_path": str(manifest_path),
                "output_dir": str(output_dir),
                "patch_size": args.patch_size,
                "stride": args.stride,
                "edge_mode": args.edge_mode,
                "target_modality": args.target_modality,
                "expected_patch_count": expected_n,
                "expected_feature_dim": args.expected_feature_dim,
            },
            "decision": decision,
            "file_summaries": file_summaries,
            "array_details": details_df.to_dict(orient="records"),
        }

        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(jsonable(summary_payload), f, indent=2, ensure_ascii=False)

        write_markdown_report(
            path=report_md,
            decision=decision,
            file_summaries=file_summaries,
            details_df=details_df,
            embeddings_dir=embeddings_dir,
            manifest_path=manifest_path,
            expected_c=args.expected_feature_dim,
        )

    log(f"      Details CSV:  {details_csv}")
    log(f"      Summary JSON: {summary_json}")
    log(f"      Report MD:    {report_md}")

    log("[5/6] Compact array summary")
    display_cols = [
        "file",
        "key",
        "shape",
        "dtype",
        "array_role",
        "upernet_usable",
        "reason",
    ]

    compact = details_df[display_cols].copy()

    with pd.option_context("display.max_rows", 200, "display.max_columns", 20, "display.width", 220):
        print(compact.to_string(index=False))

    log("")
    log("[6/6] Done")
    log("=" * 100)

    if decision["can_proceed_directly_to_upernet"]:
        log("Result: Dense CROMA features appear to be available. Next script should validate alignment.")
    else:
        log("Result: Dense CROMA features are not clearly available. Next script should extract dense CROMA features.")

    log("=" * 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect CROMA .npz embedding shapes and determine whether dense features exist for UPerNet."
    )

    parser.add_argument(
        "--instance-root",
        required=True,
        help="Dataset instance root.",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=None,
        help="Optional explicit path to metadata/croma_probing/full_embeddings.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional explicit path to the CROMA comparison manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Patch size.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=112,
        help="Patch stride.",
    )
    parser.add_argument(
        "--edge-mode",
        default="cover",
        help="Edge mode.",
    )
    parser.add_argument(
        "--target-modality",
        default="s2_s1_snap_vv_vh",
        help="Target modality to inspect as the primary UPerNet input.",
    )
    parser.add_argument(
        "--expected-patch-count",
        type=int,
        default=None,
        help="Expected number of unique patches. If omitted, inferred from manifest.",
    )
    parser.add_argument(
        "--expected-feature-dim",
        type=int,
        default=EXPECTED_FEATURE_DIM_DEFAULT,
        help="Expected CROMA feature dimension, usually 768.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and print results without writing output files.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())