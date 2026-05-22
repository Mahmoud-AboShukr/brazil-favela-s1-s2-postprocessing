"""
3_probe_croma_dense_feature_interface_224.py

Main objective
--------------
Probe whether the local CROMA implementation can expose dense spatial
features/tokens for UPerNet segmentation.

Why this script exists
----------------------
The previous inspection showed that existing CROMA .npz files contain global
patch embeddings only:

    embeddings: N x 768

For UPerNet segmentation, this is not sufficient. We need dense spatial
features/tokens such as:

    N x 196 x 768

or:

    N x 768 x 14 x 14

This script is a diagnostic probe. It does NOT extract all dense features.
It helps us discover the correct CROMA model interface before writing the full
dense feature extraction script.

What this script does
---------------------
1. Reads metadata from the target CROMA .npz file.
2. Decodes output_shapes_json, croma_model_modality, embedding_key, etc.
3. Scans local project code for CROMA model/extraction clues.
4. Checks importability of likely CROMA/PyTorch modules.
5. Optionally loads a model via --model-loader and runs a tiny dummy forward pass.
6. Registers forward hooks to detect dense token tensors or feature maps.
7. Writes CSV, JSON, and Markdown reports.

Recommended first run
---------------------
python src/splitting_strategy_experiments/3_probe_croma_dense_feature_interface_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --repo-root "C:/Users/acer/OneDrive/Desktop/UMR_espace_dev/brazil-favela-s1-s2-postprocessing" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --target-modality "s2_s1_snap_vv_vh" `
  --overwrite

Optional second run, if we identify a model loader:
--------------------------------------------------
python src/splitting_strategy_experiments/3_probe_croma_dense_feature_interface_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --repo-root "C:/Users/acer/OneDrive/Desktop/UMR_espace_dev/brazil-favela-s1-s2-postprocessing" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --target-modality "s2_s1_snap_vv_vh" `
  --model-loader "some.module.path:load_model_function" `
  --model-loader-kwargs-json "{\"model_size\":\"base\"}" `
  --overwrite
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EXPECTED_C = 768
EXPECTED_TOKENS = 196
EXPECTED_PATCH_HW = 224


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


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [jsonable(v) for v in value]

    if isinstance(value, set):
        return sorted([jsonable(v) for v in value])

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return jsonable(value.item())
        if value.size <= 20:
            return jsonable(value.tolist())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "preview": jsonable(value.reshape(-1)[:20].tolist()),
        }

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if math.isnan(v) else v

    if isinstance(value, float):
        return None if math.isnan(value) else value

    return value


def default_embedding_npz(
    instance_root: Path,
    target_modality: str,
    patch_size: int,
    stride: int,
    edge_mode: str,
) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / "full_embeddings"
        / f"croma_embeddings_{target_modality}_ps{patch_size}_st{stride}_{edge_mode}.npz"
    )


def default_manifest_path(
    instance_root: Path,
    patch_size: int,
    stride: int,
    edge_mode: str,
) -> Path:
    return (
        instance_root
        / "metadata"
        / "croma_probing"
        / f"croma_comparison_manifest_ps{patch_size}_st{stride}_{edge_mode}.csv"
    )


def default_output_dir(
    instance_root: Path,
    patch_size: int,
    stride: int,
    edge_mode: str,
) -> Path:
    return (
        instance_root
        / "metadata"
        / "splitting_strategy_experiments"
        / f"croma_dense_interface_probe_ps{patch_size}_st{stride}_{edge_mode}"
    )


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = (
        list(output_dir.glob("*.csv"))
        + list(output_dir.glob("*.json"))
        + list(output_dir.glob("*.md"))
        + list(output_dir.glob("*.txt"))
    )

    if existing and not overwrite:
        fail(
            f"Output directory already contains files:\n{output_dir}\n\n"
            f"Use --overwrite if you want to replace them."
        )


def scalar_or_preview(arr: np.ndarray, max_items: int = 10) -> Any:
    if arr.ndim == 0:
        return arr.item()

    if arr.shape == (1,):
        return arr[0].item() if hasattr(arr[0], "item") else arr[0]

    if arr.size <= max_items:
        return arr.tolist()

    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "preview": arr.reshape(-1)[:max_items].tolist(),
    }


def parse_maybe_json(value: Any) -> Any:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    # Try proper JSON first.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try Python literal representation.
    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    return text


def inspect_existing_npz_metadata(npz_path: Path) -> Dict[str, Any]:
    if not npz_path.exists():
        fail(f"Target embedding .npz file does not exist:\n{npz_path}")

    log(f"[1/7] Reading target NPZ metadata:\n{npz_path}")

    metadata: Dict[str, Any] = {
        "npz_path": str(npz_path),
        "arrays": {},
        "decoded_metadata": {},
        "dense_candidate_arrays": [],
        "global_embedding_arrays": [],
    }

    with np.load(npz_path, allow_pickle=False) as data:
        keys = list(data.keys())
        metadata["keys"] = keys

        for key in keys:
            arr = data[key]
            arr_info = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "ndim": int(arr.ndim),
                "preview_or_value": scalar_or_preview(arr),
            }

            role = classify_existing_array_shape(key, tuple(arr.shape), str(arr.dtype))
            arr_info.update(role)

            if role["array_role"] == "global_patch_embedding":
                metadata["global_embedding_arrays"].append(key)

            if role["is_dense_candidate"]:
                metadata["dense_candidate_arrays"].append(key)

            metadata["arrays"][key] = arr_info

        for key in [
            "output_shapes_json",
            "croma_model_modality",
            "embedding_key",
            "normalization",
            "model_size",
            "image_resolution",
            "modality",
        ]:
            if key in data:
                raw_value = scalar_or_preview(data[key])
                decoded = parse_maybe_json(raw_value)
                metadata["decoded_metadata"][key] = {
                    "raw": raw_value,
                    "decoded": decoded,
                }

    return metadata


def classify_existing_array_shape(
    key: str,
    shape: Tuple[int, ...],
    dtype: str,
    expected_c: int = EXPECTED_C,
    expected_tokens: int = EXPECTED_TOKENS,
) -> Dict[str, Any]:
    out = {
        "array_role": "unknown",
        "is_dense_candidate": False,
        "is_global_candidate": False,
        "upernet_usable": False,
        "reason": "",
    }

    key_norm = normalize_text(key)

    if any(tok in key_norm for tok in ["patch_id", "city", "region", "label", "modality", "created", "normalization"]):
        out["array_role"] = "metadata_or_label"
        out["reason"] = "metadata-like key"
        return out

    if len(shape) == 2 and shape[1] == expected_c:
        out["array_role"] = "global_patch_embedding"
        out["is_global_candidate"] = True
        out["upernet_usable"] = False
        out["reason"] = "N x C global embedding; not enough for dense segmentation"
        return out

    if len(shape) == 3:
        n, a, b = shape
        if a == expected_tokens and b == expected_c:
            out["array_role"] = "dense_tokens_n_tokens_c"
            out["is_dense_candidate"] = True
            out["upernet_usable"] = True
            out["reason"] = "N x 196 x 768 token sequence; likely reshapeable to N x 768 x 14 x 14"
            return out

        if a == expected_c and b == expected_tokens:
            out["array_role"] = "dense_tokens_n_c_tokens"
            out["is_dense_candidate"] = True
            out["upernet_usable"] = True
            out["reason"] = "N x 768 x 196 token sequence; likely reshapeable to N x 768 x 14 x 14"
            return out

    if len(shape) == 4:
        n, a, b, c = shape
        if a == expected_c and b > 1 and c > 1:
            out["array_role"] = "dense_feature_map_n_c_h_w"
            out["is_dense_candidate"] = True
            out["upernet_usable"] = True
            out["reason"] = "N x C x H x W feature map; directly suitable for decoder"
            return out

        if c == expected_c and a > 1 and b > 1:
            out["array_role"] = "dense_feature_map_n_h_w_c"
            out["is_dense_candidate"] = True
            out["upernet_usable"] = True
            out["reason"] = "N x H x W x C feature map; usable after transpose"
            return out

    out["reason"] = "shape does not match known dense feature patterns"
    return out


def get_python_files_to_scan(repo_root: Path) -> List[Path]:
    scan_roots = [
        repo_root / "src" / "croma_probing",
        repo_root / "src" / "splitting_strategy_experiments",
        repo_root / "src" / "upernet_croma",
    ]

    files: List[Path] = []

    for root in scan_roots:
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))

    # If croma_probing does not exist or is empty, scan src lightly.
    if not files and (repo_root / "src").exists():
        files = sorted((repo_root / "src").rglob("*.py"))

    return files


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def extract_imports_functions_classes(text: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "imports": [],
        "functions": [],
        "classes": [],
    }

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return info

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                info["imports"].append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            info["imports"].append(f"from {module} import {', '.join(names)}")

        elif isinstance(node, ast.FunctionDef):
            info["functions"].append(node.name)

        elif isinstance(node, ast.AsyncFunctionDef):
            info["functions"].append(node.name)

        elif isinstance(node, ast.ClassDef):
            info["classes"].append(node.name)

    return info


def get_keyword_snippets(
    text: str,
    keywords: Sequence[str],
    max_snippets: int = 12,
    context_lines: int = 2,
) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    snippets: List[Dict[str, Any]] = []

    for i, line in enumerate(lines):
        line_norm = line.lower()
        if any(k.lower() in line_norm for k in keywords):
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            snippets.append(
                {
                    "line": i + 1,
                    "text": "\n".join(lines[start:end]),
                }
            )

            if len(snippets) >= max_snippets:
                break

    return snippets


def scan_repo_for_croma_clues(repo_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    log("[2/7] Scanning repository code for CROMA model/extraction clues")

    files = get_python_files_to_scan(repo_root)

    keywords = [
        "croma",
        "CROMA",
        "pretrained",
        "from_pretrained",
        "load_state_dict",
        "checkpoint",
        "forward",
        "forward_features",
        "encode",
        "encoder",
        "tokens",
        "embedding_key",
        "output_shapes_json",
        "register_forward_hook",
        "return_dense",
        "return_tokens",
        "return_features",
        "intermediate",
        "get_intermediate_layers",
    ]

    rows: List[Dict[str, Any]] = []

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            rows.append(
                {
                    "file": safe_relative(path, repo_root),
                    "error": str(exc),
                    "croma_mentions": 0,
                    "important_mentions": 0,
                    "imports": "",
                    "functions": "",
                    "classes": "",
                    "snippets": "",
                }
            )
            continue

        text_lower = text.lower()
        croma_mentions = text_lower.count("croma")

        important_patterns = [
            "from_pretrained",
            "load_state_dict",
            "forward_features",
            "get_intermediate_layers",
            "return_tokens",
            "return_dense",
            "register_forward_hook",
            "embedding_key",
            "output_shapes_json",
        ]
        important_mentions = sum(text_lower.count(p.lower()) for p in important_patterns)

        if croma_mentions == 0 and important_mentions == 0:
            continue

        parsed = extract_imports_functions_classes(text)
        snippets = get_keyword_snippets(text, keywords)

        rows.append(
            {
                "file": safe_relative(path, repo_root),
                "croma_mentions": croma_mentions,
                "important_mentions": important_mentions,
                "imports": " | ".join(parsed["imports"][:25]),
                "functions": " | ".join(parsed["functions"][:40]),
                "classes": " | ".join(parsed["classes"][:20]),
                "snippets": "\n---\n".join(
                    [f"line {s['line']}:\n{s['text']}" for s in snippets]
                ),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["important_mentions", "croma_mentions", "file"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    summary = {
        "repo_root": str(repo_root),
        "python_files_scanned": len(files),
        "files_with_croma_clues": int(len(df)),
        "top_files": df["file"].head(10).tolist() if not df.empty else [],
    }

    log(f"      Python files scanned: {len(files):,}")
    log(f"      Files with CROMA clues: {len(df):,}")

    if not df.empty:
        log("      Top candidate files:")
        for file_name in df["file"].head(10).tolist():
            log(f"        - {file_name}")

    return df, summary


def check_importability(repo_root: Path) -> pd.DataFrame:
    log("[3/7] Checking importability of likely CROMA/PyTorch modules")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    candidates = [
        "torch",
        "timm",
        "einops",
        "rasterio",
        "croma",
        "CROMA",
        "models",
        "model",
        "croma_model",
        "croma.models",
        "CROMA.models",
        "src.croma_probing",
        "src.upernet_croma",
        "src.splitting_strategy_experiments",
    ]

    rows: List[Dict[str, Any]] = []

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            module_file = getattr(module, "__file__", None)
            module_version = getattr(module, "__version__", None)

            rows.append(
                {
                    "module": module_name,
                    "importable": True,
                    "file": str(module_file) if module_file else "",
                    "version": str(module_version) if module_version else "",
                    "error": "",
                }
            )

            log(f"      OK: {module_name}")

        except Exception as exc:
            rows.append(
                {
                    "module": module_name,
                    "importable": False,
                    "file": "",
                    "version": "",
                    "error": repr(exc),
                }
            )

    return pd.DataFrame(rows)


def parse_loader_string(loader: str) -> Tuple[str, str]:
    if ":" in loader:
        module_name, attr = loader.split(":", 1)
        return module_name.strip(), attr.strip()

    parts = loader.strip().split(".")
    if len(parts) < 2:
        fail(
            "--model-loader must look like 'module.path:function_name' "
            "or 'module.path.function_name'"
        )

    return ".".join(parts[:-1]), parts[-1]


def resolve_attr(root: Any, attr_path: str) -> Any:
    obj = root
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def try_load_model(
    loader: str,
    kwargs_json: Optional[str],
    checkpoint: Optional[str],
    repo_root: Path,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    log("[4/7] Trying to load model using --model-loader")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    info: Dict[str, Any] = {
        "loader": loader,
        "success": False,
        "error": "",
        "model_type": "",
        "model_repr": "",
        "kwargs": {},
    }

    kwargs: Dict[str, Any] = {}

    if kwargs_json:
        try:
            kwargs = json.loads(kwargs_json)
        except Exception as exc:
            fail(f"Could not parse --model-loader-kwargs-json: {exc}")

    if checkpoint:
        # We include several common names but only if the function accepts **kwargs
        # or the specific argument. If it does not, the error will be visible.
        kwargs.setdefault("checkpoint", checkpoint)
        kwargs.setdefault("checkpoint_path", checkpoint)
        kwargs.setdefault("ckpt_path", checkpoint)
        kwargs.setdefault("pretrained_path", checkpoint)

    info["kwargs"] = kwargs

    try:
        module_name, attr_path = parse_loader_string(loader)
        module = importlib.import_module(module_name)
        obj = resolve_attr(module, attr_path)

        if inspect.isclass(obj):
            model = obj(**kwargs)
        elif callable(obj):
            model = obj(**kwargs)
        else:
            fail(f"Resolved loader object is not callable: {loader}")

        # Some loader functions return dictionaries or tuples.
        if isinstance(model, dict):
            for key in ["model", "encoder", "net", "module"]:
                if key in model:
                    model = model[key]
                    break

        if isinstance(model, (tuple, list)):
            for item in model:
                if hasattr(item, "forward") or hasattr(item, "__call__"):
                    model = item
                    break

        info["success"] = True
        info["model_type"] = str(type(model))
        info["model_repr"] = repr(model)[:5000]

        log(f"      Model loaded successfully: {type(model)}")

        return model, info

    except Exception as exc:
        info["error"] = traceback.format_exc()
        warn(f"Model loading failed: {exc}")
        return None, info


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def choose_device(device_arg: str) -> str:
    import torch

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return device_arg


def tensor_shape_summary(obj: Any, max_depth: int = 3) -> Any:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(obj, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
            "requires_grad": bool(obj.requires_grad),
        }

    if isinstance(obj, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    if max_depth <= 0:
        return str(type(obj))

    if isinstance(obj, dict):
        return {
            str(k): tensor_shape_summary(v, max_depth=max_depth - 1)
            for k, v in list(obj.items())[:30]
        }

    if isinstance(obj, (list, tuple)):
        return [
            tensor_shape_summary(v, max_depth=max_depth - 1)
            for v in list(obj)[:30]
        ]

    return {
        "type": str(type(obj)),
        "repr": repr(obj)[:500],
    }


def is_leaf_module(module: Any) -> bool:
    try:
        return len(list(module.children())) == 0
    except Exception:
        return False


def module_name_priority(name: str, module: Any) -> int:
    name_norm = name.lower()
    cls_norm = module.__class__.__name__.lower()

    score = 0

    tokens = [
        "block",
        "encoder",
        "transformer",
        "attn",
        "attention",
        "mlp",
        "norm",
        "patch",
        "embed",
        "proj",
        "head",
        "fusion",
        "cross",
        "vit",
        "backbone",
    ]

    for token in tokens:
        if token in name_norm or token in cls_norm:
            score += 1

    return score


def classify_tensor_shape_for_dense_probe(
    shape: Sequence[int],
    batch_size: int,
    expected_c: int,
    expected_tokens: int,
) -> Dict[str, Any]:
    out = {
        "dense_candidate": False,
        "candidate_type": "not_dense_candidate",
        "reason": "",
    }

    shape = list(shape)

    if not shape:
        out["reason"] = "empty shape"
        return out

    if shape[0] != batch_size:
        out["reason"] = "first dimension does not match batch size"
        return out

    if len(shape) == 2:
        if shape[1] == expected_c:
            out["candidate_type"] = "global_b_c"
            out["reason"] = "B x C global embedding"
        else:
            out["reason"] = "2D tensor but not B x expected_C"
        return out

    if len(shape) == 3:
        b, a, c = shape

        if a in [expected_tokens, expected_tokens + 1] and c == expected_c:
            out["dense_candidate"] = True
            out["candidate_type"] = "tokens_b_n_c"
            out["reason"] = "B x tokens x C, likely dense ViT tokens"
            return out

        if a == expected_c and c in [expected_tokens, expected_tokens + 1]:
            out["dense_candidate"] = True
            out["candidate_type"] = "tokens_b_c_n"
            out["reason"] = "B x C x tokens, likely dense ViT tokens"
            return out

        if c in [expected_c, 384, 512, 1024] and a > 4:
            out["dense_candidate"] = True
            out["candidate_type"] = "possible_tokens_b_n_c"
            out["reason"] = "B x N x C-like tensor; possible token sequence"
            return out

    if len(shape) == 4:
        b, a, h, w = shape

        if a in [expected_c, 384, 512, 1024] and h > 1 and w > 1:
            out["dense_candidate"] = True
            out["candidate_type"] = "feature_map_b_c_h_w"
            out["reason"] = "B x C x H x W feature map"
            return out

        if w in [expected_c, 384, 512, 1024] and a > 1 and h > 1:
            out["dense_candidate"] = True
            out["candidate_type"] = "feature_map_b_h_w_c"
            out["reason"] = "B x H x W x C feature map"
            return out

    out["reason"] = "not a recognized dense feature shape"
    return out


@dataclass
class HookRecorder:
    batch_size: int
    expected_c: int
    expected_tokens: int
    max_records: int

    def __post_init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.handles: List[Any] = []
        self.seen: set = set()

    def hook_fn(self, module_name: str, module_type: str) -> Callable:
        def _hook(module: Any, inputs: Tuple[Any, ...], output: Any) -> None:
            if len(self.records) >= self.max_records:
                return

            tensors = self.extract_tensors(output)

            for tensor_index, tensor in enumerate(tensors):
                shape = list(tensor.shape)
                key = (module_name, module_type, tuple(shape), str(tensor.dtype))

                if key in self.seen:
                    continue

                self.seen.add(key)

                classification = classify_tensor_shape_for_dense_probe(
                    shape=shape,
                    batch_size=self.batch_size,
                    expected_c=self.expected_c,
                    expected_tokens=self.expected_tokens,
                )

                self.records.append(
                    {
                        "module_name": module_name,
                        "module_type": module_type,
                        "tensor_index": tensor_index,
                        "shape": str(tuple(shape)),
                        "shape_json": json.dumps(shape),
                        "dtype": str(tensor.dtype),
                        "device": str(tensor.device),
                        **classification,
                    }
                )

                if len(self.records) >= self.max_records:
                    break

        return _hook

    def extract_tensors(self, obj: Any) -> List[Any]:
        import torch

        tensors: List[Any] = []

        if isinstance(obj, torch.Tensor):
            return [obj]

        if isinstance(obj, dict):
            for value in obj.values():
                tensors.extend(self.extract_tensors(value))
            return tensors

        if isinstance(obj, (list, tuple)):
            for value in obj:
                tensors.extend(self.extract_tensors(value))
            return tensors

        return tensors

    def register(self, model: Any, max_modules: int) -> None:
        named_modules = list(model.named_modules())

        candidates = []
        for name, module in named_modules:
            if name == "":
                continue

            score = module_name_priority(name, module)

            if score > 0 or is_leaf_module(module):
                candidates.append((score, name, module))

        candidates = sorted(candidates, key=lambda x: (-x[0], x[1]))

        registered = 0
        for _, name, module in candidates:
            if registered >= max_modules:
                break

            try:
                handle = module.register_forward_hook(
                    self.hook_fn(name, module.__class__.__name__)
                )
                self.handles.append(handle)
                registered += 1
            except Exception:
                continue

    def remove(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass

        self.handles = []


def make_dummy_inputs(
    batch_size: int,
    patch_size: int,
    device: str,
    dtype: str = "float32",
) -> Dict[str, Any]:
    import torch

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    s2 = torch.randn(batch_size, 12, patch_size, patch_size, device=device, dtype=torch_dtype)
    sar = torch.randn(batch_size, 2, patch_size, patch_size, device=device, dtype=torch_dtype)

    return {
        "s2": s2,
        "sar": sar,
        "optical": s2,
        "SAR": sar,
        "sentinel2": s2,
        "sentinel1": sar,
    }


def try_forward_patterns(model: Any, inputs: Dict[str, Any]) -> Tuple[bool, str, Any, List[Dict[str, Any]]]:
    import torch

    s2 = inputs["s2"]
    sar = inputs["sar"]

    attempts: List[Tuple[str, Callable[[], Any]]] = [
        ("model(s2=s2, sar=sar)", lambda: model(s2=s2, sar=sar)),
        ("model(s2=s2, SAR=sar)", lambda: model(s2=s2, SAR=sar)),
        ("model(optical=s2, sar=sar)", lambda: model(optical=s2, sar=sar)),
        ("model(optical=s2, SAR=sar)", lambda: model(optical=s2, SAR=sar)),
        ("model(sentinel2=s2, sentinel1=sar)", lambda: model(sentinel2=s2, sentinel1=sar)),
        ("model(S2=s2, S1=sar)", lambda: model(S2=s2, S1=sar)),
        ("model(s2, sar)", lambda: model(s2, sar)),
        ("model({'s2': s2, 'sar': sar})", lambda: model({"s2": s2, "sar": sar})),
        ("model({'optical': s2, 'sar': sar})", lambda: model({"optical": s2, "sar": sar})),
        ("model({'SAR': sar, 'optical': s2})", lambda: model({"SAR": sar, "optical": s2})),
        ("model(s2)", lambda: model(s2)),
        ("model(sar)", lambda: model(sar)),
    ]

    errors: List[Dict[str, Any]] = []

    for name, fn in attempts:
        try:
            with torch.no_grad():
                output = fn()

            return True, name, output, errors

        except Exception as exc:
            errors.append(
                {
                    "attempt": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return False, "", None, errors


def run_optional_forward_probe(
    model: Any,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    log("[5/7] Running optional dummy forward-pass probe")

    if model is None:
        log("      No model was loaded, so forward-pass probing is skipped.")
        return pd.DataFrame(), {
            "attempted": False,
            "success": False,
            "reason": "No model loader was provided or model loading failed.",
        }

    if not torch_available():
        warn("PyTorch is not importable, so forward-pass probing is skipped.")
        return pd.DataFrame(), {
            "attempted": False,
            "success": False,
            "reason": "PyTorch is not importable.",
        }

    import torch

    device = choose_device(args.device)

    try:
        model = model.to(device)
    except Exception:
        warn("Could not move model with model.to(device). Continuing anyway.")

    try:
        model.eval()
    except Exception:
        warn("Could not call model.eval(). Continuing anyway.")

    dummy_inputs = make_dummy_inputs(
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        device=device,
        dtype=args.dummy_dtype,
    )

    recorder = HookRecorder(
        batch_size=args.batch_size,
        expected_c=args.expected_feature_dim,
        expected_tokens=args.expected_tokens,
        max_records=args.max_hook_records,
    )

    recorder.register(model, max_modules=args.max_hook_modules)

    try:
        success, attempt_name, output, errors = try_forward_patterns(model, dummy_inputs)

        output_summary = tensor_shape_summary(output)

        forward_summary = {
            "attempted": True,
            "success": success,
            "successful_attempt": attempt_name,
            "output_summary": output_summary,
            "failed_attempts": errors,
            "device": device,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "expected_feature_dim": args.expected_feature_dim,
            "expected_tokens": args.expected_tokens,
        }

    finally:
        recorder.remove()

    hooks_df = pd.DataFrame(recorder.records)

    if not hooks_df.empty:
        hooks_df = hooks_df.sort_values(
            ["dense_candidate", "candidate_type", "module_name"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    if forward_summary["success"]:
        log(f"      Forward pass succeeded using: {forward_summary['successful_attempt']}")
        dense_count = int(hooks_df["dense_candidate"].sum()) if not hooks_df.empty else 0
        log(f"      Dense candidate tensors found by hooks: {dense_count}")
    else:
        warn("Forward pass did not succeed with any automatic call pattern.")

    return hooks_df, forward_summary


def build_final_decision(
    npz_metadata: Dict[str, Any],
    repo_scan_df: pd.DataFrame,
    import_df: pd.DataFrame,
    hooks_df: pd.DataFrame,
    forward_summary: Dict[str, Any],
    model_load_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    dense_in_npz = bool(npz_metadata.get("dense_candidate_arrays"))
    global_in_npz = bool(npz_metadata.get("global_embedding_arrays"))

    repo_has_clues = not repo_scan_df.empty
    torch_importable = bool(
        not import_df.empty
        and ((import_df["module"] == "torch") & (import_df["importable"] == True)).any()
    )

    forward_success = bool(forward_summary.get("success", False))
    hook_dense = bool(not hooks_df.empty and hooks_df["dense_candidate"].fillna(False).any())

    if dense_in_npz:
        recommendation = "dense_features_already_present_in_npz"
        conclusion = (
            "The existing target NPZ already appears to contain dense features. "
            "Proceed to dense feature validation."
        )
        next_step = "validate_dense_features"

    elif forward_success and hook_dense:
        recommendation = "dense_interface_found_write_full_extractor_next"
        conclusion = (
            "The dummy forward pass succeeded and hooks detected dense candidate tensors. "
            "Proceed to writing the full dense CROMA feature extraction script."
        )
        next_step = "write_full_dense_extractor"

    elif forward_success and not hook_dense:
        recommendation = "forward_works_but_dense_tokens_not_detected"
        conclusion = (
            "The model forward pass succeeded, but the automatic hooks did not detect dense token "
            "or feature-map tensors. Inspect the forward output and hook CSV to identify the correct layer."
        )
        next_step = "manual_layer_selection"

    elif model_load_info and model_load_info.get("success") and not forward_success:
        recommendation = "model_loaded_forward_signature_unknown"
        conclusion = (
            "The model loaded, but automatic dummy forward patterns failed. "
            "We need to inspect the model forward signature or reuse the exact call from the existing extractor."
        )
        next_step = "inspect_forward_signature"

    elif repo_has_clues and torch_importable:
        recommendation = "static_clues_found_need_model_loader"
        conclusion = (
            "The existing NPZ is global-only, but repository code contains CROMA-related clues. "
            "Use the report to identify the previous CROMA loader/extractor function, then rerun this script with --model-loader."
        )
        next_step = "rerun_with_model_loader"

    else:
        recommendation = "need_locate_croma_implementation"
        conclusion = (
            "The existing NPZ is global-only and no usable dense interface was found automatically. "
            "We need to locate the exact CROMA implementation or previous extraction script."
        )
        next_step = "locate_croma_code"

    return {
        "dense_candidate_arrays_in_existing_npz": npz_metadata.get("dense_candidate_arrays", []),
        "global_embedding_arrays_in_existing_npz": npz_metadata.get("global_embedding_arrays", []),
        "repo_has_croma_clues": repo_has_clues,
        "torch_importable": torch_importable,
        "forward_success": forward_success,
        "hook_dense_candidates_found": hook_dense,
        "recommendation": recommendation,
        "next_step": next_step,
        "conclusion": conclusion,
    }


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    npz_metadata: Dict[str, Any],
    repo_scan_df: pd.DataFrame,
    import_df: pd.DataFrame,
    hooks_df: pd.DataFrame,
    model_load_info: Optional[Dict[str, Any]],
    forward_summary: Dict[str, Any],
    final_decision: Dict[str, Any],
) -> None:
    lines: List[str] = []

    lines.append("# CROMA Dense Feature Interface Probe")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "Probe whether the local CROMA implementation can expose dense spatial "
        "features/tokens suitable for UPerNet segmentation."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Instance root: `{args.instance_root}`")
    lines.append(f"- Repository root: `{args.repo_root}`")
    lines.append(f"- Target modality: `{args.target_modality}`")
    lines.append(f"- Patch size: `{args.patch_size}`")
    lines.append(f"- Stride: `{args.stride}`")
    lines.append(f"- Edge mode: `{args.edge_mode}`")
    lines.append(f"- Target NPZ: `{npz_metadata['npz_path']}`")
    lines.append("")
    lines.append("## Final Decision")
    lines.append("")
    lines.append(f"- Recommendation: `{final_decision['recommendation']}`")
    lines.append(f"- Next step: `{final_decision['next_step']}`")
    lines.append(f"- Forward success: `{final_decision['forward_success']}`")
    lines.append(f"- Dense candidates found by hooks: `{final_decision['hook_dense_candidates_found']}`")
    lines.append("")
    lines.append(final_decision["conclusion"])
    lines.append("")

    lines.append("## Existing Target NPZ Metadata")
    lines.append("")
    lines.append(f"- Keys: `{', '.join(npz_metadata.get('keys', []))}`")
    lines.append(f"- Global embedding arrays: `{npz_metadata.get('global_embedding_arrays', [])}`")
    lines.append(f"- Dense candidate arrays: `{npz_metadata.get('dense_candidate_arrays', [])}`")
    lines.append("")
    lines.append("| Key | Shape | Dtype | Role | UPerNet usable? | Reason |")
    lines.append("|---|---|---|---|---:|---|")

    for key, info in npz_metadata["arrays"].items():
        lines.append(
            f"| `{key}` "
            f"| `{tuple(info['shape'])}` "
            f"| `{info['dtype']}` "
            f"| `{info.get('array_role', '')}` "
            f"| `{info.get('upernet_usable', False)}` "
            f"| {str(info.get('reason', '')).replace('|', '\\|')} |"
        )

    lines.append("")
    lines.append("## Decoded Saved Metadata")
    lines.append("")

    decoded_metadata = npz_metadata.get("decoded_metadata", {})
    if decoded_metadata:
        for key, item in decoded_metadata.items():
            lines.append(f"### `{key}`")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(jsonable(item.get("decoded")), indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
    else:
        lines.append("No decoded metadata was available.")
        lines.append("")

    lines.append("## Importability Check")
    lines.append("")
    if import_df.empty:
        lines.append("No import checks were run.")
    else:
        lines.append("| Module | Importable | Version | File/Error |")
        lines.append("|---|---:|---|---|")
        for _, row in import_df.iterrows():
            file_or_error = row["file"] if row["importable"] else row["error"]
            lines.append(
                f"| `{row['module']}` | `{row['importable']}` | `{row['version']}` | `{str(file_or_error).replace('|', '\\|')}` |"
            )

    lines.append("")
    lines.append("## Repository CROMA Clues")
    lines.append("")

    if repo_scan_df.empty:
        lines.append("No CROMA-related source-code clues were found.")
    else:
        lines.append("| File | CROMA mentions | Important mentions | Functions | Classes |")
        lines.append("|---|---:|---:|---|---|")
        for _, row in repo_scan_df.head(20).iterrows():
            lines.append(
                f"| `{row['file']}` "
                f"| {row['croma_mentions']} "
                f"| {row['important_mentions']} "
                f"| `{str(row['functions'])[:250]}` "
                f"| `{str(row['classes'])[:250]}` |"
            )

    lines.append("")
    lines.append("## Model Loading")
    lines.append("")
    if model_load_info is None:
        lines.append("No `--model-loader` was provided, so no model was loaded.")
    else:
        lines.append("```json")
        compact_model_info = dict(model_load_info)
        if compact_model_info.get("model_repr"):
            compact_model_info["model_repr"] = compact_model_info["model_repr"][:2000]
        if compact_model_info.get("error"):
            compact_model_info["error"] = compact_model_info["error"][-4000:]
        lines.append(json.dumps(jsonable(compact_model_info), indent=2, ensure_ascii=False))
        lines.append("```")

    lines.append("")
    lines.append("## Forward Probe")
    lines.append("")
    lines.append("```json")
    compact_forward = dict(forward_summary)
    if compact_forward.get("failed_attempts"):
        compact_forward["failed_attempts"] = compact_forward["failed_attempts"][:10]
    lines.append(json.dumps(jsonable(compact_forward), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Hook Dense Candidate Tensors")
    lines.append("")

    if hooks_df.empty:
        lines.append("No hook tensors were recorded.")
    else:
        dense_df = hooks_df[hooks_df["dense_candidate"] == True].copy()
        if dense_df.empty:
            lines.append("Hooks recorded tensors, but none matched dense feature/token patterns.")
        else:
            lines.append("| Module | Type | Shape | Candidate type | Reason |")
            lines.append("|---|---|---|---|---|")
            for _, row in dense_df.head(50).iterrows():
                lines.append(
                    f"| `{row['module_name']}` "
                    f"| `{row['module_type']}` "
                    f"| `{row['shape']}` "
                    f"| `{row['candidate_type']}` "
                    f"| {str(row['reason']).replace('|', '\\|')} |"
                )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "If the final recommendation is `static_clues_found_need_model_loader`, open the repo scan CSV/Markdown "
        "and find the previous CROMA model-loading function. Then rerun this script with `--model-loader`."
    )
    lines.append("")
    lines.append(
        "If dense hook candidates are found, the next script should extract those tensors for all patches "
        "and save them as dense features aligned with `patch_ids`."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    instance_root = Path(args.instance_root)
    repo_root = Path(args.repo_root).resolve()

    target_npz = Path(args.embedding_npz) if args.embedding_npz else default_embedding_npz(
        instance_root=instance_root,
        target_modality=args.target_modality,
        patch_size=args.patch_size,
        stride=args.stride,
        edge_mode=args.edge_mode,
    )

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
    log("CROMA Dense Feature Interface Probe")
    log("=" * 100)
    log(f"Instance root:      {instance_root}")
    log(f"Repository root:    {repo_root}")
    log(f"Target modality:    {args.target_modality}")
    log(f"Target NPZ:         {target_npz}")
    log(f"Manifest:           {manifest_path}")
    log(f"Output dir:         {output_dir}")
    log(f"Patch size:         {args.patch_size}")
    log(f"Stride:             {args.stride}")
    log(f"Edge mode:          {args.edge_mode}")
    log(f"Expected C:         {args.expected_feature_dim}")
    log(f"Expected tokens:    {args.expected_tokens}")
    log("=" * 100)

    ensure_output_dir(output_dir, overwrite=args.overwrite)

    npz_metadata = inspect_existing_npz_metadata(target_npz)

    repo_scan_df, repo_scan_summary = scan_repo_for_croma_clues(repo_root)

    import_df = check_importability(repo_root)

    model = None
    model_load_info: Optional[Dict[str, Any]] = None

    if args.model_loader:
        model, model_load_info = try_load_model(
            loader=args.model_loader,
            kwargs_json=args.model_loader_kwargs_json,
            checkpoint=args.checkpoint,
            repo_root=repo_root,
        )
    else:
        log("[4/7] No --model-loader provided, skipping model loading.")

    if args.skip_forward:
        hooks_df = pd.DataFrame()
        forward_summary = {
            "attempted": False,
            "success": False,
            "reason": "--skip-forward was used.",
        }
    else:
        hooks_df, forward_summary = run_optional_forward_probe(model, args)

    final_decision = build_final_decision(
        npz_metadata=npz_metadata,
        repo_scan_df=repo_scan_df,
        import_df=import_df,
        hooks_df=hooks_df,
        forward_summary=forward_summary,
        model_load_info=model_load_info,
    )

    log("[6/7] Writing outputs")

    repo_scan_csv = output_dir / f"croma_dense_probe_repo_scan_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    import_csv = output_dir / f"croma_dense_probe_import_check_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    hook_csv = output_dir / f"croma_dense_probe_hook_shapes_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    summary_json = output_dir / f"croma_dense_probe_summary_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.json"
    report_md = output_dir / f"croma_dense_probe_report_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.md"

    if not args.dry_run:
        repo_scan_df.to_csv(repo_scan_csv, index=False)
        import_df.to_csv(import_csv, index=False)
        hooks_df.to_csv(hook_csv, index=False)

        payload = {
            "configuration": {
                "instance_root": str(instance_root),
                "repo_root": str(repo_root),
                "target_modality": args.target_modality,
                "target_npz": str(target_npz),
                "manifest_path": str(manifest_path),
                "output_dir": str(output_dir),
                "patch_size": args.patch_size,
                "stride": args.stride,
                "edge_mode": args.edge_mode,
                "expected_feature_dim": args.expected_feature_dim,
                "expected_tokens": args.expected_tokens,
                "model_loader": args.model_loader,
                "checkpoint": args.checkpoint,
            },
            "npz_metadata": npz_metadata,
            "repo_scan_summary": repo_scan_summary,
            "model_load_info": model_load_info,
            "forward_summary": forward_summary,
            "final_decision": final_decision,
            "repo_scan_rows": repo_scan_df.to_dict(orient="records"),
            "import_check_rows": import_df.to_dict(orient="records"),
            "hook_rows": hooks_df.to_dict(orient="records"),
        }

        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)

        write_markdown_report(
            path=report_md,
            args=args,
            npz_metadata=npz_metadata,
            repo_scan_df=repo_scan_df,
            import_df=import_df,
            hooks_df=hooks_df,
            model_load_info=model_load_info,
            forward_summary=forward_summary,
            final_decision=final_decision,
        )

    log(f"      Repo scan CSV: {repo_scan_csv}")
    log(f"      Import CSV:    {import_csv}")
    log(f"      Hook CSV:      {hook_csv}")
    log(f"      Summary JSON:  {summary_json}")
    log(f"      Report MD:     {report_md}")

    log("[7/7] Final decision")
    log(f"      Recommendation: {final_decision['recommendation']}")
    log(f"      Next step:      {final_decision['next_step']}")
    log(f"      Conclusion:     {final_decision['conclusion']}")

    log("=" * 100)
    log("Done.")
    log("=" * 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether local CROMA can expose dense spatial features for UPerNet."
    )

    parser.add_argument(
        "--instance-root",
        required=True,
        help="Dataset instance root.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root. Use your brazil-favela-s1-s2-postprocessing path.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional explicit CROMA comparison manifest path.",
    )
    parser.add_argument(
        "--embedding-npz",
        default=None,
        help="Optional explicit target CROMA .npz file.",
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
        help="Target modality.",
    )
    parser.add_argument(
        "--expected-feature-dim",
        type=int,
        default=EXPECTED_C,
        help="Expected dense feature dimension, usually 768.",
    )
    parser.add_argument(
        "--expected-tokens",
        type=int,
        default=EXPECTED_TOKENS,
        help="Expected token count for 224/16 patches, usually 196.",
    )

    parser.add_argument(
        "--model-loader",
        default=None,
        help=(
            "Optional model loader in form 'module.path:function_name'. "
            "If omitted, the script only performs static probing."
        ),
    )
    parser.add_argument(
        "--model-loader-kwargs-json",
        default=None,
        help="Optional JSON kwargs passed to the model loader.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint/pretrained path passed to the model loader kwargs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for optional forward probe: auto, cpu, cuda.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size for optional forward probe.",
    )
    parser.add_argument(
        "--dummy-dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Dummy tensor dtype for optional forward probe.",
    )
    parser.add_argument(
        "--max-hook-modules",
        type=int,
        default=400,
        help="Maximum number of modules to hook during optional forward probe.",
    )
    parser.add_argument(
        "--max-hook-records",
        type=int,
        default=1000,
        help="Maximum hook tensor records to keep.",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Skip optional forward pass even if a model loader is provided.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing output files.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())