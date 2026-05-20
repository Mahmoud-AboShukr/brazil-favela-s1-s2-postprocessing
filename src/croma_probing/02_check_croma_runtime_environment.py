#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_check_croma_runtime_environment.py

Check whether the local runtime is ready for CROMA embedding extraction.

This script does NOT extract embeddings.
It validates:

    1. Python environment.
    2. Required Python packages.
    3. PyTorch and CUDA availability.
    4. NVIDIA GPU information if available.
    5. CROMA official route availability:
        - use_croma.py
        - PretrainedCROMA
        - pretrained weights
    6. Optional TorchGeo CROMA availability.
    7. Instance C CROMA manifest validity.
    8. Optional small model smoke test.

Official CROMA route:
    The official repository uses:
        from use_croma import PretrainedCROMA

    Sentinel-1 input:
        [N, 2, H, W] = VV, VH

    Sentinel-2 input:
        [N, 12, H, W]

Recommended for our fair RTC-vs-SNAP-GRD comparison:
    SNAP-GRD -> use only VV/VH bands 1 and 2
    RTC      -> use VV/VH bands 1 and 2

Example basic check:

python src/croma_probing/02_check_croma_runtime_environment.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --overwrite

Example with official CROMA repo and weights:

python src/croma_probing/02_check_croma_runtime_environment.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --croma-repo "C:/Users/acer/OneDrive/Desktop/UMR_espace_dev/CROMA" `
  --weights-path "D:/models/CROMA/CROMA_base.pt" `
  --image-resolution 224 `
  --model-size base `
  --run-smoke-test `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


# ---------------------------------------------------------------------
# CSV / JSON / Markdown
# ---------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        fail(f"CSV does not exist: {path_to_str(path)}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        fail(f"CSV is empty: {path_to_str(path)}")

    return rows


def read_json_optional(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    check_rows: List[Dict[str, object]],
    package_rows: List[Dict[str, object]],
    overwrite: bool,
) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# CROMA runtime environment check")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Error count: `{summary['error_count']}`")
    lines.append(f"- Warning count: `{summary['warning_count']}`")
    lines.append(f"- Python executable: `{summary['python']['executable']}`")
    lines.append(f"- Python version: `{summary['python']['version']}`")
    lines.append(f"- Platform: `{summary['python']['platform']}`")
    lines.append("")

    lines.append("## PyTorch / CUDA")
    lines.append("")
    torch_info = summary["torch"]
    lines.append(f"- Torch available: `{torch_info['torch_available']}`")
    lines.append(f"- Torch version: `{torch_info['torch_version']}`")
    lines.append(f"- CUDA available: `{torch_info['cuda_available']}`")
    lines.append(f"- Torch CUDA version: `{torch_info['torch_cuda_version']}`")
    lines.append(f"- cuDNN version: `{torch_info['cudnn_version']}`")
    lines.append(f"- GPU count: `{torch_info['gpu_count']}`")
    lines.append(f"- Selected device: `{torch_info['selected_device']}`")
    lines.append(f"- Selected GPU name: `{torch_info['selected_gpu_name']}`")
    lines.append(f"- Selected GPU total memory GB: `{torch_info['selected_gpu_total_memory_gb']}`")
    lines.append("")

    lines.append("## CROMA route")
    lines.append("")
    croma = summary["croma"]
    lines.append(f"- Official CROMA repo: `{croma['croma_repo']}`")
    lines.append(f"- use_croma.py found: `{croma['use_croma_py_found']}`")
    lines.append(f"- PretrainedCROMA importable: `{croma['pretrained_croma_importable']}`")
    lines.append(f"- Weights path: `{croma['weights_path']}`")
    lines.append(f"- Weights found: `{croma['weights_found']}`")
    lines.append(f"- Smoke test requested: `{croma['smoke_test_requested']}`")
    lines.append(f"- Smoke test status: `{croma['smoke_test_status']}`")
    lines.append(f"- Smoke test details: `{croma['smoke_test_details']}`")
    lines.append("")

    lines.append("## Manifest")
    lines.append("")
    manifest = summary["manifest"]
    lines.append(f"- Comparison manifest: `{manifest['comparison_manifest_csv']}`")
    lines.append(f"- Rows: `{manifest['rows']}`")
    lines.append(f"- Expected rows: `{manifest['expected_rows']}`")
    lines.append(f"- Modalities: `{';'.join(manifest['modalities'])}`")
    lines.append(f"- SNAP uses VV/VH only: `{manifest['snap_uses_vv_vh_only']}`")
    lines.append(f"- RTC uses VV/VH only: `{manifest['rtc_uses_vv_vh_only']}`")
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
    lines.append("## Package checks")
    lines.append("")
    lines.append("| package | required | status | version | details |")
    lines.append("|---|---|---|---|---|")
    for row in package_rows:
        lines.append(
            f"| {row['package']} | "
            f"{row['required']} | "
            f"{row['status']} | "
            f"{row['version']} | "
            f"{row['details']} |"
        )

    lines.append("")
    lines.append("## Next step")
    lines.append("")
    if summary["status"] == "passed":
        lines.append("The runtime is ready for the first CROMA embedding extraction test.")
    else:
        lines.append("Fix the failed checks before running CROMA embedding extraction.")

    lines.append("")
    lines.append("Recommended next script after this check passes:")
    lines.append("")
    lines.append("```text")
    lines.append("src/croma_probing/03_extract_croma_embeddings_smoke_test_224.py")
    lines.append("```")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------

def add_check(
    rows: List[Dict[str, object]],
    *,
    check_name: str,
    severity: str,
    status: str,
    details: str = "",
) -> None:
    rows.append(
        {
            "check_name": check_name,
            "severity": severity,
            "status": status,
            "details": details,
        }
    )


def package_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "")
        return str(version)
    except Exception:
        return ""


def check_import(module_name: str) -> Tuple[bool, str, str]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "")
        return True, str(version), ""
    except Exception as exc:
        return False, "", repr(exc)


def check_packages() -> List[Dict[str, object]]:
    required_packages = [
        ("torch", True),
        ("numpy", True),
        ("pandas", True),
        ("rasterio", True),
        ("sklearn", True),
        ("einops", True),
        ("tqdm", True),
    ]

    optional_packages = [
        ("torchvision", False),
        ("h5py", False),
        ("matplotlib", False),
        ("umap", False),
        ("torchgeo", False),
    ]

    rows: List[Dict[str, object]] = []

    for module_name, required in required_packages + optional_packages:
        ok, version, details = check_import(module_name)

        rows.append(
            {
                "package": module_name,
                "required": required,
                "status": "ok" if ok else "missing",
                "version": version,
                "details": details,
            }
        )

    return rows


def check_torch(device_index: int) -> Dict[str, object]:
    info: Dict[str, object] = {
        "torch_available": False,
        "torch_version": "",
        "cuda_available": False,
        "torch_cuda_version": "",
        "cudnn_version": "",
        "gpu_count": 0,
        "selected_device": "cpu",
        "selected_gpu_name": "",
        "selected_gpu_total_memory_gb": "",
        "selected_gpu_free_memory_gb": "",
        "selected_gpu_allocated_memory_gb": "",
        "selected_gpu_reserved_memory_gb": "",
        "device_check_error": "",
    }

    try:
        import torch

        info["torch_available"] = True
        info["torch_version"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = "" if torch.version.cuda is None else str(torch.version.cuda)

        try:
            info["cudnn_version"] = "" if torch.backends.cudnn.version() is None else str(torch.backends.cudnn.version())
        except Exception:
            info["cudnn_version"] = ""

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            info["gpu_count"] = int(gpu_count)

            if device_index < 0 or device_index >= gpu_count:
                info["device_check_error"] = f"Requested CUDA device index {device_index}, but gpu_count={gpu_count}."
                return info

            torch.cuda.set_device(device_index)
            device = torch.device(f"cuda:{device_index}")
            props = torch.cuda.get_device_properties(device)

            info["selected_device"] = f"cuda:{device_index}"
            info["selected_gpu_name"] = str(props.name)
            info["selected_gpu_total_memory_gb"] = round(float(props.total_memory) / (1024 ** 3), 3)

            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                info["selected_gpu_free_memory_gb"] = round(float(free_bytes) / (1024 ** 3), 3)
            except Exception:
                info["selected_gpu_free_memory_gb"] = ""

            info["selected_gpu_allocated_memory_gb"] = round(float(torch.cuda.memory_allocated(device)) / (1024 ** 3), 3)
            info["selected_gpu_reserved_memory_gb"] = round(float(torch.cuda.memory_reserved(device)) / (1024 ** 3), 3)

    except Exception as exc:
        info["device_check_error"] = repr(exc)

    return info


def run_nvidia_smi() -> Dict[str, object]:
    result = {
        "available": False,
        "returncode": "",
        "stdout": "",
        "stderr": "",
    }

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

        result["available"] = completed.returncode == 0
        result["returncode"] = completed.returncode
        result["stdout"] = completed.stdout.strip()
        result["stderr"] = completed.stderr.strip()

    except Exception as exc:
        result["stderr"] = repr(exc)

    return result


def resolve_weights_path(args: argparse.Namespace) -> Optional[Path]:
    if args.weights_path is not None:
        return args.weights_path

    if args.weights_dir is None:
        return None

    name = "CROMA_base.pt" if args.model_size == "base" else "CROMA_large.pt"
    return args.weights_dir / name


def import_pretrained_croma(croma_repo: Optional[Path]) -> Tuple[bool, str]:
    try:
        if croma_repo is not None:
            if not croma_repo.exists():
                return False, f"CROMA repo path does not exist: {path_to_str(croma_repo)}"

            use_croma_path = croma_repo / "use_croma.py"

            if not use_croma_path.exists():
                return False, f"use_croma.py not found in CROMA repo: {path_to_str(use_croma_path)}"

            repo_str = str(croma_repo.resolve())
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

        from use_croma import PretrainedCROMA  # noqa: F401

        return True, "Imported PretrainedCROMA from use_croma."

    except Exception as exc:
        return False, repr(exc)


def check_torchgeo_croma() -> Dict[str, object]:
    info = {
        "torchgeo_available": False,
        "torchgeo_version": "",
        "croma_available": False,
        "details": "",
    }

    try:
        import torchgeo

        info["torchgeo_available"] = True
        info["torchgeo_version"] = getattr(torchgeo, "__version__", "")

        try:
            import torchgeo.models as models

            names = dir(models)
            croma_names = [name for name in names if "croma" in name.lower()]
            info["croma_available"] = len(croma_names) > 0
            info["details"] = ";".join(croma_names)

        except Exception as exc:
            info["details"] = repr(exc)

    except Exception as exc:
        info["details"] = repr(exc)

    return info


def run_croma_smoke_test(
    *,
    croma_repo: Optional[Path],
    weights_path: Path,
    model_size: str,
    image_resolution: int,
    device: str,
    batch_size: int,
) -> Tuple[str, str]:
    try:
        ok, details = import_pretrained_croma(croma_repo)

        if not ok:
            return "failed", f"Could not import PretrainedCROMA: {details}"

        if not weights_path.exists():
            return "failed", f"Weights path does not exist: {path_to_str(weights_path)}"

        import torch
        from use_croma import PretrainedCROMA

        if device.startswith("cuda") and not torch.cuda.is_available():
            return "failed", "CUDA device requested but torch.cuda.is_available() is False."

        model = PretrainedCROMA(
            pretrained_path=str(weights_path),
            size=model_size,
            modality="both",
            image_resolution=int(image_resolution),
        ).to(device)

        model.eval()

        sar = torch.rand(batch_size, 2, image_resolution, image_resolution, device=device)
        optical = torch.rand(batch_size, 12, image_resolution, image_resolution, device=device)

        with torch.no_grad():
            outputs = model(SAR_images=sar, optical_images=optical)

        keys = sorted(list(outputs.keys()))

        expected_keys = [
            "SAR_GAP",
            "SAR_encodings",
            "joint_GAP",
            "joint_encodings",
            "optical_GAP",
            "optical_encodings",
        ]

        missing = [key for key in expected_keys if key not in outputs]

        if missing:
            return "failed", f"Smoke test ran, but expected output keys are missing: {missing}. Actual keys: {keys}"

        shape_parts = []

        for key in expected_keys:
            value = outputs[key]
            if hasattr(value, "shape"):
                shape_parts.append(f"{key}={tuple(value.shape)}")

        return "passed", "; ".join(shape_parts)

    except Exception:
        return "failed", traceback.format_exc().replace("\n", " | ")


def validate_manifest(
    comparison_manifest_csv: Path,
    manifest_json: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    rows = read_csv_rows(comparison_manifest_csv)
    payload = read_json_optional(manifest_json)

    modalities = sorted(set(r["modality"] for r in rows))
    modality_counter = Counter(r["modality"] for r in rows)

    expected_modalities = [
        "s2",
        "s1_snap_vv_vh",
        "s1_rtc_vv_vh",
        "s2_s1_snap_vv_vh",
        "s2_s1_rtc_vv_vh",
    ]

    snap_rows = [r for r in rows if r["sar_variant"] == "snap_grd"]
    rtc_rows = [r for r in rows if r["sar_variant"] == "rtc"]

    snap_uses_vv_vh_only = all(
        r["sar_band_indices"] == "1;2"
        and r["sar_channel_names"] == "VV;VH"
        and r["snap_ignored_band_indices"] == "3"
        for r in snap_rows
    )

    rtc_uses_vv_vh_only = all(
        r["sar_band_indices"] == "1;2"
        and r["sar_channel_names"] == "VV;VH"
        for r in rtc_rows
    )

    return {
        "comparison_manifest_csv": path_to_str(comparison_manifest_csv),
        "manifest_json": path_to_str(manifest_json),
        "rows": len(rows),
        "expected_rows": int(args.expected_total_patches) * len(expected_modalities),
        "modalities": modalities,
        "expected_modalities": expected_modalities,
        "modality_counts": dict(modality_counter),
        "json_loaded": bool(payload),
        "snap_uses_vv_vh_only": bool(snap_uses_vv_vh_only),
        "rtc_uses_vv_vh_only": bool(rtc_uses_vv_vh_only),
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    *,
    instance_root: Path,
    comparison_manifest_csv: Path,
    manifest_json: Path,
    validation_json: Path,
    check_rows: List[Dict[str, object]],
    package_rows: List[Dict[str, object]],
    torch_info: Dict[str, object],
    nvidia_smi: Dict[str, object],
    croma_info: Dict[str, object],
    torchgeo_info: Dict[str, object],
    manifest_info: Dict[str, object],
    args: argparse.Namespace,
    output_paths: Dict[str, Path],
) -> Dict[str, object]:
    error_count = sum(1 for row in check_rows if row["severity"] == "error" and row["status"] != "passed")
    warning_count = sum(1 for row in check_rows if row["severity"] == "warning" and row["status"] != "passed")

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instance_root": path_to_str(instance_root),
        "comparison_manifest_csv": path_to_str(comparison_manifest_csv),
        "manifest_json": path_to_str(manifest_json),
        "validation_json": path_to_str(validation_json),
        "status": "passed" if error_count == 0 else "failed",
        "error_count": error_count,
        "warning_count": warning_count,
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "version_info": list(sys.version_info[:3]),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": package_rows,
        "torch": torch_info,
        "nvidia_smi": nvidia_smi,
        "croma": croma_info,
        "torchgeo": torchgeo_info,
        "manifest": manifest_info,
        "parameters": {
            "croma_repo": path_to_str(args.croma_repo),
            "weights_dir": path_to_str(args.weights_dir),
            "weights_path": path_to_str(args.weights_path),
            "model_size": args.model_size,
            "image_resolution": args.image_resolution,
            "device_index": args.device_index,
            "run_smoke_test": bool(args.run_smoke_test),
            "smoke_test_batch_size": args.smoke_test_batch_size,
            "require_cuda": bool(args.require_cuda),
            "require_croma": bool(args.require_croma),
            "require_weights": bool(args.require_weights),
        },
        "outputs": {key: path_to_str(value) for key, value in output_paths.items()},
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check runtime environment for CROMA embedding extraction."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--comparison-manifest-csv",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/croma_comparison_manifest_ps<patch-size>_st<stride>_<edge-mode>.csv",
    )

    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/croma_comparison_manifest_ps<patch-size>_st<stride>_<edge-mode>.json",
    )

    parser.add_argument(
        "--validation-json",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/instance_C_patches/patch_metadata_validation_ps<patch-size>_st<stride>_<edge-mode>.json",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing.",
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
        "--expected-total-patches",
        type=int,
        default=12699,
        help="Expected total patches. Default: 12699.",
    )

    parser.add_argument(
        "--croma-repo",
        type=Path,
        default=None,
        help="Path to official CROMA repo containing use_croma.py.",
    )

    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory containing CROMA_base.pt or CROMA_large.pt.",
    )

    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Explicit path to CROMA weights.",
    )

    parser.add_argument(
        "--model-size",
        choices=["base", "large"],
        default="base",
        help="CROMA model size. Default: base.",
    )

    parser.add_argument(
        "--image-resolution",
        type=int,
        default=224,
        help="Image resolution for our patch inputs. Default: 224.",
    )

    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index. Default: 0.",
    )

    parser.add_argument(
        "--run-smoke-test",
        action="store_true",
        help="Instantiate PretrainedCROMA and run one synthetic forward pass.",
    )

    parser.add_argument(
        "--smoke-test-batch-size",
        type=int,
        default=1,
        help="Batch size for synthetic CROMA smoke test. Default: 1.",
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA is not available.",
    )

    parser.add_argument(
        "--require-croma",
        action="store_true",
        help="Fail if official PretrainedCROMA cannot be imported.",
    )

    parser.add_argument(
        "--require-weights",
        action="store_true",
        help="Fail if CROMA weights are not found.",
    )

    parser.add_argument(
        "--no-fail-on-warning",
        action="store_true",
        help="Warnings do not affect exit status anyway; this flag is kept for clarity.",
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
    output_dir: Path = args.output_dir or (instance_root / "metadata" / "croma_probing")

    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    comparison_manifest_csv: Path = args.comparison_manifest_csv or (
        output_dir / f"croma_comparison_manifest_{stem}.csv"
    )

    manifest_json: Path = args.manifest_json or (
        output_dir / f"croma_comparison_manifest_{stem}.json"
    )

    validation_json: Path = args.validation_json or (
        instance_root / "metadata" / "instance_C_patches" / f"patch_metadata_validation_{stem}.json"
    )

    checks_csv = output_dir / f"croma_runtime_environment_checks_{stem}.csv"
    packages_csv = output_dir / f"croma_runtime_environment_packages_{stem}.csv"
    json_path = output_dir / f"croma_runtime_environment_{stem}.json"
    md_path = output_dir / f"croma_runtime_environment_{stem}.md"

    output_paths = {
        "checks_csv": checks_csv,
        "packages_csv": packages_csv,
        "json": json_path,
        "markdown": md_path,
    }

    log("STEP", "Checking CROMA runtime environment.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Manifest CSV:  {path_to_str(comparison_manifest_csv)}")
    log("INFO", f"CROMA repo:    {path_to_str(args.croma_repo)}")
    log("INFO", f"Weights path:  {path_to_str(args.weights_path)}")
    log("INFO", f"Weights dir:   {path_to_str(args.weights_dir)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    check_rows: List[Dict[str, object]] = []

    package_rows = check_packages()

    missing_required = [
        row["package"]
        for row in package_rows
        if row["required"] is True and row["status"] != "ok"
    ]

    add_check(
        check_rows,
        check_name="required_python_packages",
        severity="error",
        status="passed" if not missing_required else "failed",
        details="" if not missing_required else "Missing: " + ";".join(missing_required),
    )

    torch_info = check_torch(device_index=int(args.device_index))

    add_check(
        check_rows,
        check_name="torch_available",
        severity="error",
        status="passed" if torch_info["torch_available"] else "failed",
        details=str(torch_info.get("device_check_error", "")),
    )

    if args.require_cuda:
        cuda_status = bool(torch_info["cuda_available"])
        cuda_severity = "error"
    else:
        cuda_status = bool(torch_info["cuda_available"])
        cuda_severity = "warning"

    add_check(
        check_rows,
        check_name="cuda_available",
        severity=cuda_severity,
        status="passed" if cuda_status else "failed",
        details=(
            f"selected_device={torch_info['selected_device']}; "
            f"gpu={torch_info['selected_gpu_name']}; "
            f"torch_cuda={torch_info['torch_cuda_version']}"
        ),
    )

    nvidia_smi = run_nvidia_smi()

    add_check(
        check_rows,
        check_name="nvidia_smi_available",
        severity="warning",
        status="passed" if nvidia_smi["available"] else "failed",
        details=nvidia_smi["stdout"] or nvidia_smi["stderr"],
    )

    croma_repo = args.croma_repo
    use_croma_py_found = False

    if croma_repo is not None:
        use_croma_py_found = (croma_repo / "use_croma.py").exists()
    else:
        spec = importlib.util.find_spec("use_croma")
        use_croma_py_found = spec is not None

    croma_import_ok, croma_import_details = import_pretrained_croma(croma_repo)

    croma_severity = "error" if args.require_croma else "warning"

    add_check(
        check_rows,
        check_name="official_croma_pretrained_import",
        severity=croma_severity,
        status="passed" if croma_import_ok else "failed",
        details=croma_import_details,
    )

    weights_path = resolve_weights_path(args)
    weights_found = weights_path is not None and weights_path.exists()

    weights_severity = "error" if args.require_weights or args.run_smoke_test else "warning"

    add_check(
        check_rows,
        check_name="croma_weights_found",
        severity=weights_severity,
        status="passed" if weights_found else "failed",
        details="" if weights_path is None else path_to_str(weights_path),
    )

    torchgeo_info = check_torchgeo_croma()

    add_check(
        check_rows,
        check_name="torchgeo_croma_available_optional",
        severity="warning",
        status="passed" if torchgeo_info["croma_available"] else "failed",
        details=torchgeo_info["details"],
    )

    manifest_info = validate_manifest(comparison_manifest_csv, manifest_json, args)

    add_check(
        check_rows,
        check_name="comparison_manifest_row_count",
        severity="error",
        status="passed" if manifest_info["rows"] == manifest_info["expected_rows"] else "failed",
        details=f"rows={manifest_info['rows']}, expected={manifest_info['expected_rows']}",
    )

    add_check(
        check_rows,
        check_name="snap_uses_vv_vh_only",
        severity="error",
        status="passed" if manifest_info["snap_uses_vv_vh_only"] else "failed",
        details="SNAP-GRD should use sar_band_indices=1;2 and ignore band 3.",
    )

    add_check(
        check_rows,
        check_name="rtc_uses_vv_vh_only",
        severity="error",
        status="passed" if manifest_info["rtc_uses_vv_vh_only"] else "failed",
        details="RTC should use sar_band_indices=1;2.",
    )

    validation_payload = read_json_optional(validation_json)
    validation_status = validation_payload.get("validation_status", "")

    add_check(
        check_rows,
        check_name="patch_metadata_validation_passed",
        severity="error",
        status="passed" if validation_status == "passed" else "failed",
        details=f"validation_status={validation_status}",
    )

    smoke_test_status = "not_requested"
    smoke_test_details = ""

    if args.run_smoke_test:
        if weights_path is None:
            smoke_test_status = "failed"
            smoke_test_details = "No weights path provided or resolved."
        else:
            selected_device = str(torch_info["selected_device"])
            smoke_test_status, smoke_test_details = run_croma_smoke_test(
                croma_repo=croma_repo,
                weights_path=weights_path,
                model_size=str(args.model_size),
                image_resolution=int(args.image_resolution),
                device=selected_device,
                batch_size=int(args.smoke_test_batch_size),
            )

        add_check(
            check_rows,
            check_name="croma_model_smoke_test",
            severity="error",
            status=smoke_test_status,
            details=smoke_test_details[:500],
        )

    croma_info = {
        "croma_repo": path_to_str(croma_repo),
        "use_croma_py_found": bool(use_croma_py_found),
        "pretrained_croma_importable": bool(croma_import_ok),
        "pretrained_croma_import_details": croma_import_details,
        "weights_path": "" if weights_path is None else path_to_str(weights_path),
        "weights_found": bool(weights_found),
        "model_size": args.model_size,
        "image_resolution": args.image_resolution,
        "smoke_test_requested": bool(args.run_smoke_test),
        "smoke_test_status": smoke_test_status,
        "smoke_test_details": smoke_test_details,
    }

    summary = build_summary(
        instance_root=instance_root,
        comparison_manifest_csv=comparison_manifest_csv,
        manifest_json=manifest_json,
        validation_json=validation_json,
        check_rows=check_rows,
        package_rows=package_rows,
        torch_info=torch_info,
        nvidia_smi=nvidia_smi,
        croma_info=croma_info,
        torchgeo_info=torchgeo_info,
        manifest_info=manifest_info,
        args=args,
        output_paths=output_paths,
    )

    log("STEP", "Writing runtime check outputs.")

    write_csv(
        checks_csv,
        check_rows,
        overwrite=bool(args.overwrite),
        fieldnames=["check_name", "severity", "status", "details"],
    )

    write_csv(
        packages_csv,
        package_rows,
        overwrite=bool(args.overwrite),
        fieldnames=["package", "required", "status", "version", "details"],
    )

    write_json(json_path, summary, overwrite=bool(args.overwrite))
    write_markdown(md_path, summary, check_rows, package_rows, overwrite=bool(args.overwrite))

    log("OK", f"Wrote checks CSV:   {path_to_str(checks_csv)}")
    log("OK", f"Wrote packages CSV: {path_to_str(packages_csv)}")
    log("OK", f"Wrote JSON:         {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:     {path_to_str(md_path)}")

    log("STEP", "Final runtime summary.")
    log("OK" if summary["status"] == "passed" else "ERROR", f"Status: {summary['status']}")
    log("OK", f"Errors: {summary['error_count']}")
    log("OK", f"Warnings: {summary['warning_count']}")
    log("OK", f"Torch: {torch_info['torch_version']}")
    log("OK" if torch_info["cuda_available"] else "WARN", f"CUDA available: {torch_info['cuda_available']}")
    log("OK", f"Selected device: {torch_info['selected_device']}")
    log("OK", f"Selected GPU: {torch_info['selected_gpu_name']}")
    log("OK" if croma_import_ok else "WARN", f"PretrainedCROMA importable: {croma_import_ok}")
    log("OK" if weights_found else "WARN", f"CROMA weights found: {weights_found}")
    log("OK", f"Manifest rows: {manifest_info['rows']}")
    log("OK", f"SNAP uses VV/VH only: {manifest_info['snap_uses_vv_vh_only']}")
    log("OK", f"RTC uses VV/VH only: {manifest_info['rtc_uses_vv_vh_only']}")

    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()