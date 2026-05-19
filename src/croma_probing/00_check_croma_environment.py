#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
00_check_croma_environment.py

Check whether the local environment can run CROMA on Instance C patches.

This script verifies:

1. Python / PyTorch / CUDA availability.
2. TorchGeo CROMA availability.
3. Optional official CROMA repository availability through use_croma.py.
4. Whether 224x224 CROMA inputs are accepted.
5. Whether a real Instance C patch can be read from the metadata CSV.
6. What output keys and tensor shapes are returned.

Important project convention:

Instance C S1_READY has 3 bands:
    1. VV_dB
    2. VH_dB
    3. VV_minus_VH_dB

CROMA expects SAR input with 2 channels, so this script uses only:
    VV_dB and VH_dB

CROMA expects optical input with 12 channels, so this script uses all 12 S2 bands.

Example PowerShell command:

python src/croma_probing/00_check_croma_environment.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --device auto `
  --run-forward `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "[ERROR] PyTorch is required but is not installed.\n"
        "Install PyTorch first, preferably with CUDA support if you plan to use the GPU.\n\n"
        f"Original error: {exc}"
    )

try:
    import rasterio
    from rasterio.windows import Window
except ImportError as exc:
    raise SystemExit(
        "[ERROR] rasterio is required but is not installed.\n"
        "Install it first, for example:\n"
        "    pip install rasterio\n\n"
        f"Original error: {exc}"
    )


# ---------------------------------------------------------------------
# Logging helpers
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


def get_package_version(package_name: str) -> str:
    try:
        import importlib.metadata as metadata
        return metadata.version(package_name)
    except Exception:
        return "not_installed_or_unknown"


def parse_int(row: Dict[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except Exception as exc:
        raise ValueError(
            f"Could not parse integer column '{key}' from row with patch_id="
            f"{row.get('patch_id', '<unknown>')}"
        ) from exc


# ---------------------------------------------------------------------
# CSV and report I/O
# ---------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        fail(f"Input CSV does not exist: {path_to_str(path)}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        fail(f"Input CSV is empty: {path_to_str(path)}")

    return rows


def write_json(path: Path, payload: Dict[str, Any], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(path: Path, report: Dict[str, Any], overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    lines: List[str] = []

    lines.append("# CROMA environment check")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Created UTC: `{report['created_utc']}`")
    lines.append(f"- Status: `{report['status']}`")
    lines.append(f"- Device used: `{report['device']['selected_device']}`")
    lines.append(f"- Instance root: `{report['paths']['instance_root']}`")
    lines.append(f"- Metadata CSV: `{report['paths']['metadata_csv']}`")
    lines.append(f"- Image size: `{report['parameters']['image_size']}`")
    lines.append(f"- Run forward: `{report['parameters']['run_forward']}`")
    lines.append("")

    lines.append("## System")
    lines.append("")
    system = report["system"]
    lines.append(f"- Python: `{system['python_version']}`")
    lines.append(f"- Platform: `{system['platform']}`")
    lines.append(f"- Executable: `{system['python_executable']}`")
    lines.append("")

    lines.append("## Packages")
    lines.append("")
    lines.append("| package | version/status |")
    lines.append("|---|---|")
    for name, value in report["packages"].items():
        lines.append(f"| {name} | `{value}` |")
    lines.append("")

    lines.append("## CUDA")
    lines.append("")
    device = report["device"]
    lines.append(f"- CUDA available: `{device['cuda_available']}`")
    lines.append(f"- CUDA device count: `{device['cuda_device_count']}`")
    lines.append(f"- CUDA devices: `{device['cuda_devices']}`")
    lines.append("")

    lines.append("## Dataset sample")
    lines.append("")
    sample = report["dataset_sample"]
    lines.append(f"- Sample status: `{sample['status']}`")
    lines.append(f"- Patch ID: `{sample.get('patch_id', '')}`")
    lines.append(f"- City: `{sample.get('city', '')}`")
    lines.append(f"- Region: `{sample.get('region', '')}`")
    lines.append(f"- S2 shape: `{sample.get('s2_shape', '')}`")
    lines.append(f"- S1 full shape: `{sample.get('s1_full_shape', '')}`")
    lines.append(f"- S1 CROMA shape: `{sample.get('s1_croma_shape', '')}`")
    lines.append(f"- Label shape: `{sample.get('label_shape', '')}`")
    lines.append("")

    lines.append("## CROMA routes")
    lines.append("")
    lines.append("| route | import status | model status | forward status | notes |")
    lines.append("|---|---|---|---|---|")

    for route in report["croma_routes"]:
        lines.append(
            f"| {route['route_name']} | "
            f"`{route['import_status']}` | "
            f"`{route['model_status']}` | "
            f"`{route['forward_status']}` | "
            f"{route.get('notes', '')} |"
        )

    lines.append("")

    lines.append("## Output tensor shapes")
    lines.append("")

    any_outputs = False

    for route in report["croma_routes"]:
        if route.get("output_description"):
            any_outputs = True
            lines.append(f"### {route['route_name']}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(route["output_description"], indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")

    if not any_outputs:
        lines.append("- No forward outputs available.")
        lines.append("")

    lines.append("## Recommendations")
    lines.append("")

    for item in report["recommendations"]:
        lines.append(f"- {item}")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Tensor and dataset helpers
# ---------------------------------------------------------------------

def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            fail("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    fail(f"Unsupported device: {requested}")


def describe_object(obj: Any) -> Any:
    """
    Recursively describe tensors / dicts / lists.

    This avoids dumping huge tensors into JSON.
    """

    if torch.is_tensor(obj):
        return {
            "type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
            "requires_grad": bool(obj.requires_grad),
        }

    if isinstance(obj, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    if isinstance(obj, dict):
        return {
            str(key): describe_object(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [describe_object(value) for value in obj]

    if obj is None:
        return None

    return {
        "type": type(obj).__name__,
        "repr": repr(obj)[:500],
    }


def normalize_per_channel_0_1(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Robust per-channel normalization to [0, 1] for a single patch.

    Input:
        C x H x W

    This is only for environment smoke testing.
    Later embedding extraction should use a project-level normalization policy.
    """

    array = array.astype(np.float32, copy=False)
    output = np.zeros_like(array, dtype=np.float32)

    for c in range(array.shape[0]):
        band = array[c]
        finite = np.isfinite(band)

        if not finite.any():
            output[c] = 0.0
            continue

        valid_values = band[finite]

        lo = np.percentile(valid_values, 2)
        hi = np.percentile(valid_values, 98)

        if abs(float(hi - lo)) < eps:
            output[c] = 0.0
            continue

        norm = (band - lo) / (hi - lo)
        norm = np.clip(norm, 0.0, 1.0)
        norm[~finite] = 0.0

        output[c] = norm.astype(np.float32)

    return output


def read_sample_patch_from_metadata(
    metadata_csv: Path,
    sample_mode: str,
) -> Dict[str, Any]:
    rows = read_csv_rows(metadata_csv)

    if sample_mode == "first":
        selected = rows[0]
    elif sample_mode == "first_positive":
        positives = [
            row for row in rows
            if str(row.get("patch_label_binary", "0")).strip() in {"1", "True", "true"}
        ]
        if not positives:
            fail("sample_mode='first_positive' requested but no positive patches were found.")
        selected = positives[0]
    else:
        fail(f"Unsupported sample mode: {sample_mode}")

    s2_path = Path(selected["source_s2_path"])
    s1_path = Path(selected["source_s1_snap_grd_path"])
    label_path = Path(selected["source_label_path"])

    if not s2_path.exists():
        fail(f"Sample S2 path does not exist: {path_to_str(s2_path)}")
    if not s1_path.exists():
        fail(f"Sample S1 path does not exist: {path_to_str(s1_path)}")
    if not label_path.exists():
        fail(f"Sample label path does not exist: {path_to_str(label_path)}")

    row_start = parse_int(selected, "row_start")
    col_start = parse_int(selected, "col_start")
    height = parse_int(selected, "height")
    width = parse_int(selected, "width")

    window = Window(
        col_off=col_start,
        row_off=row_start,
        width=width,
        height=height,
    )

    with rasterio.open(s2_path) as s2_src:
        s2 = s2_src.read(window=window, masked=True).filled(0).astype(np.float32)

    with rasterio.open(s1_path) as s1_src:
        s1_full = s1_src.read(window=window, masked=True).filled(0).astype(np.float32)

    with rasterio.open(label_path) as label_src:
        label = label_src.read(1, window=window, masked=True).filled(0).astype(np.uint8)

    if s2.shape[0] != 12:
        fail(f"Expected S2 sample to have 12 bands, got shape {s2.shape}")

    if s1_full.shape[0] < 2:
        fail(f"Expected S1 sample to have at least 2 bands, got shape {s1_full.shape}")

    # CROMA uses VV and VH only.
    s1_croma = s1_full[:2, :, :]

    s2_norm = normalize_per_channel_0_1(s2)
    s1_norm = normalize_per_channel_0_1(s1_croma)

    return {
        "status": "ok",
        "row": selected,
        "patch_id": selected["patch_id"],
        "city": selected["city"],
        "region": selected["region"],
        "s2": s2_norm,
        "s1_full": s1_full,
        "s1_croma": s1_norm,
        "label": label,
        "s2_shape": list(s2.shape),
        "s1_full_shape": list(s1_full.shape),
        "s1_croma_shape": list(s1_croma.shape),
        "label_shape": list(label.shape),
    }


def make_random_inputs(
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_sar = torch.rand(batch_size, 2, image_size, image_size, dtype=torch.float32, device=device)
    x_optical = torch.rand(batch_size, 12, image_size, image_size, dtype=torch.float32, device=device)

    return x_sar, x_optical


def make_real_sample_inputs(
    sample: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s1 = sample["s1_croma"]
    s2 = sample["s2"]

    x_sar = torch.from_numpy(s1).unsqueeze(0).float().to(device)
    x_optical = torch.from_numpy(s2).unsqueeze(0).float().to(device)

    return x_sar, x_optical


# ---------------------------------------------------------------------
# CROMA route tests
# ---------------------------------------------------------------------

def test_torchgeo_croma(
    *,
    device: torch.device,
    image_size: int,
    batch_size: int,
    run_forward: bool,
    use_real_sample: bool,
    sample: Optional[Dict[str, Any]],
    use_pretrained: bool,
) -> Dict[str, Any]:
    route: Dict[str, Any] = {
        "route_name": "torchgeo.models.croma_base",
        "import_status": "not_started",
        "model_status": "not_started",
        "forward_status": "not_started",
        "output_description": None,
        "notes": "",
        "error": "",
    }

    try:
        torchgeo_models = importlib.import_module("torchgeo.models")
        croma_base = getattr(torchgeo_models, "croma_base")
        route["import_status"] = "ok"
    except Exception as exc:
        route["import_status"] = "failed"
        route["model_status"] = "skipped"
        route["forward_status"] = "skipped"
        route["error"] = repr(exc)
        route["notes"] = "Install TorchGeo >= 0.7 if you want to use this route."
        return route

    try:
        weights = None

        if use_pretrained:
            CROMABase_Weights = getattr(torchgeo_models, "CROMABase_Weights")
            weights = getattr(CROMABase_Weights, "DEFAULT", None)

            if weights is None:
                route["notes"] += " CROMABase_Weights.DEFAULT not found; using weights=None."

        model = croma_base(
            weights=weights,
            modalities=["sar", "optical"],
            image_size=image_size,
        )

        model = model.to(device)
        model.eval()

        route["model_status"] = "ok"
    except Exception as exc:
        route["model_status"] = "failed"
        route["forward_status"] = "skipped"
        route["error"] = traceback.format_exc()
        return route

    if not run_forward:
        route["forward_status"] = "skipped"
        route["notes"] += " Forward pass skipped by user."
        return route

    try:
        if use_real_sample and sample is not None:
            x_sar, x_optical = make_real_sample_inputs(sample, device)
        else:
            x_sar, x_optical = make_random_inputs(batch_size, image_size, device)

        with torch.no_grad():
            try:
                outputs = model(x_sar=x_sar, x_optical=x_optical)
            except TypeError:
                # Some interfaces may use positional arguments.
                outputs = model(x_sar, x_optical)

        route["forward_status"] = "ok"
        route["output_description"] = describe_object(outputs)

    except Exception as exc:
        route["forward_status"] = "failed"
        route["error"] = traceback.format_exc()

    return route


def test_official_croma_use_croma(
    *,
    device: torch.device,
    image_size: int,
    batch_size: int,
    run_forward: bool,
    use_real_sample: bool,
    sample: Optional[Dict[str, Any]],
    official_croma_dir: Optional[Path],
    pretrained_path: Optional[Path],
    official_size: str,
) -> Dict[str, Any]:
    route: Dict[str, Any] = {
        "route_name": "official_CROMA_use_croma.PretrainedCROMA",
        "import_status": "not_started",
        "model_status": "not_started",
        "forward_status": "not_started",
        "output_description": None,
        "notes": "",
        "error": "",
    }

    if official_croma_dir is not None:
        if not official_croma_dir.exists():
            route["import_status"] = "failed"
            route["model_status"] = "skipped"
            route["forward_status"] = "skipped"
            route["error"] = f"official_croma_dir does not exist: {path_to_str(official_croma_dir)}"
            return route

        sys.path.insert(0, str(official_croma_dir))

    try:
        use_croma = importlib.import_module("use_croma")
        PretrainedCROMA = getattr(use_croma, "PretrainedCROMA")
        route["import_status"] = "ok"
    except Exception as exc:
        route["import_status"] = "failed"
        route["model_status"] = "skipped"
        route["forward_status"] = "skipped"
        route["error"] = repr(exc)
        route["notes"] = (
            "Official CROMA route is optional. It requires use_croma.py from "
            "https://github.com/antofuller/CROMA and pretrained weights."
        )
        return route

    if pretrained_path is None or not pretrained_path.exists():
        route["model_status"] = "skipped"
        route["forward_status"] = "skipped"
        route["notes"] = (
            "use_croma.py was importable, but --pretrained-path was not provided "
            "or does not exist, so official PretrainedCROMA loading was skipped."
        )
        return route

    try:
        model = PretrainedCROMA(
            pretrained_path=str(pretrained_path),
            size=official_size,
            modality="both",
            image_resolution=image_size,
        )

        model = model.to(device)
        model.eval()

        route["model_status"] = "ok"
    except Exception:
        route["model_status"] = "failed"
        route["forward_status"] = "skipped"
        route["error"] = traceback.format_exc()
        return route

    if not run_forward:
        route["forward_status"] = "skipped"
        route["notes"] += " Forward pass skipped by user."
        return route

    try:
        if use_real_sample and sample is not None:
            x_sar, x_optical = make_real_sample_inputs(sample, device)
        else:
            x_sar, x_optical = make_random_inputs(batch_size, image_size, device)

        with torch.no_grad():
            outputs = model(SAR_images=x_sar, optical_images=x_optical)

        route["forward_status"] = "ok"
        route["output_description"] = describe_object(outputs)

    except Exception:
        route["forward_status"] = "failed"
        route["error"] = traceback.format_exc()

    return route


# ---------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------

def build_recommendations(report: Dict[str, Any]) -> List[str]:
    recommendations: List[str] = []

    torchgeo_route = next(
        (r for r in report["croma_routes"] if r["route_name"] == "torchgeo.models.croma_base"),
        None,
    )

    official_route = next(
        (r for r in report["croma_routes"] if r["route_name"] == "official_CROMA_use_croma.PretrainedCROMA"),
        None,
    )

    if torchgeo_route and torchgeo_route["import_status"] != "ok":
        recommendations.append(
            "TorchGeo CROMA is not available. Try installing TorchGeo, for example: pip install torchgeo"
        )

    if torchgeo_route and torchgeo_route["model_status"] == "ok" and torchgeo_route["forward_status"] == "ok":
        recommendations.append(
            "TorchGeo CROMA forward pass works. You can use this route for the next embedding-extraction script."
        )

    if torchgeo_route and torchgeo_route["model_status"] == "ok" and torchgeo_route["forward_status"] == "failed":
        recommendations.append(
            "TorchGeo CROMA loaded but forward failed. Check image_size, input tensor shapes, CUDA memory, and TorchGeo version."
        )

    if official_route and official_route["import_status"] != "ok":
        recommendations.append(
            "Official use_croma.py route is not available. This is optional if TorchGeo CROMA works."
        )

    if not report["device"]["cuda_available"]:
        recommendations.append(
            "CUDA is not available. CROMA may run slowly on CPU; install a CUDA-compatible PyTorch build if using the RTX GPU."
        )

    if int(report["parameters"]["image_size"]) != 224:
        recommendations.append(
            "The current project patch size is 224. Use --image-size 224 for the final CROMA workflow unless debugging."
        )

    if not recommendations:
        recommendations.append("Environment looks ready for the next CROMA embedding extraction step.")

    return recommendations


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether CROMA can run on Instance C 224x224 patches."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
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
        "--metadata-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit patch metadata CSV. "
            "Default: <instance-root>/metadata/instance_C_patches/"
            "patch_metadata_ps<patch-size>_st<stride>_<edge-mode>.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. "
            "Default: <instance-root>/qc/croma_environment_check"
        ),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use. Default: auto.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="CROMA image size. Must be a multiple of 8. Default: 224.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for random forward test. Default: 1.",
    )

    parser.add_argument(
        "--sample-mode",
        choices=["first", "first_positive"],
        default="first_positive",
        help="Which real metadata patch to sample. Default: first_positive.",
    )

    parser.add_argument(
        "--run-forward",
        action="store_true",
        help="Actually run a CROMA forward pass.",
    )

    parser.add_argument(
        "--use-random-input",
        action="store_true",
        help="Use random tensors instead of a real patch from the metadata CSV.",
    )

    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help=(
            "Try to load TorchGeo pretrained CROMA weights. "
            "This may download weights if not cached."
        ),
    )

    parser.add_argument(
        "--official-croma-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to a clone of the official CROMA repository containing use_croma.py."
        ),
    )

    parser.add_argument(
        "--pretrained-path",
        type=Path,
        default=None,
        help=(
            "Optional path to official CROMA weights, e.g. CROMA_base.pt. "
            "Needed only for the official use_croma.py route."
        ),
    )

    parser.add_argument(
        "--official-size",
        choices=["base", "large"],
        default="base",
        help="Official CROMA size when using use_croma.py. Default: base.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previous JSON/Markdown check reports.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.image_size % 8 != 0:
        fail(f"--image-size must be a multiple of 8 for CROMA. Got {args.image_size}.")

    instance_root: Path = args.instance_root

    metadata_csv: Path = args.metadata_csv or (
        instance_root
        / "metadata"
        / "instance_C_patches"
        / f"patch_metadata_ps{args.patch_size}_st{args.stride}_{args.edge_mode}.csv"
    )

    output_dir: Path = args.output_dir or (
        instance_root
        / "qc"
        / "croma_environment_check"
    )

    output_json = output_dir / "croma_environment_check.json"
    output_md = output_dir / "croma_environment_check.md"

    log("STEP", "Checking CROMA environment.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Metadata CSV:  {path_to_str(metadata_csv)}")
    log("INFO", f"Output dir:    {path_to_str(output_dir)}")

    if not instance_root.exists():
        fail(f"Instance root does not exist: {path_to_str(instance_root)}")

    device = choose_device(args.device)

    log("INFO", f"Selected device: {device}")

    sample_report: Dict[str, Any] = {
        "status": "not_loaded",
    }

    sample: Optional[Dict[str, Any]] = None

    if args.use_random_input:
        sample_report = {
            "status": "skipped_using_random_input",
        }
        log("WARN", "Using random input; real metadata patch reading is skipped.")
    else:
        log("STEP", "Reading one real patch from metadata.")
        sample = read_sample_patch_from_metadata(metadata_csv, args.sample_mode)

        sample_report = {
            "status": "ok",
            "patch_id": sample["patch_id"],
            "city": sample["city"],
            "region": sample["region"],
            "s2_shape": sample["s2_shape"],
            "s1_full_shape": sample["s1_full_shape"],
            "s1_croma_shape": sample["s1_croma_shape"],
            "label_shape": sample["label_shape"],
        }

        log(
            "OK",
            f"Read sample patch {sample['patch_id']} "
            f"from {sample['city']} / {sample['region']}",
        )
        log("OK", f"S2 shape: {sample['s2_shape']}")
        log("OK", f"S1 full shape: {sample['s1_full_shape']}")
        log("OK", f"S1 CROMA shape: {sample['s1_croma_shape']}")
        log("OK", f"Label shape: {sample['label_shape']}")

    report: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "paths": {
            "instance_root": path_to_str(instance_root),
            "metadata_csv": path_to_str(metadata_csv),
            "output_json": path_to_str(output_json),
            "output_md": path_to_str(output_md),
        },
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.stride,
            "edge_mode": args.edge_mode,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "run_forward": bool(args.run_forward),
            "use_random_input": bool(args.use_random_input),
            "use_pretrained": bool(args.use_pretrained),
            "sample_mode": args.sample_mode,
        },
        "system": {
            "python_version": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            "torch": torch.__version__,
            "torchvision": get_package_version("torchvision"),
            "torchgeo": get_package_version("torchgeo"),
            "einops": get_package_version("einops"),
            "timm": get_package_version("timm"),
            "rasterio": get_package_version("rasterio"),
            "numpy": np.__version__,
        },
        "device": {
            "selected_device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_devices": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        },
        "dataset_sample": sample_report,
        "croma_routes": [],
        "recommendations": [],
    }

    log("STEP", "Testing TorchGeo CROMA route.")

    torchgeo_route = test_torchgeo_croma(
        device=device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        run_forward=args.run_forward,
        use_real_sample=not args.use_random_input,
        sample=sample,
        use_pretrained=args.use_pretrained,
    )

    report["croma_routes"].append(torchgeo_route)

    log(
        "INFO",
        f"TorchGeo route: import={torchgeo_route['import_status']}, "
        f"model={torchgeo_route['model_status']}, "
        f"forward={torchgeo_route['forward_status']}",
    )

    log("STEP", "Testing optional official CROMA route.")

    official_route = test_official_croma_use_croma(
        device=device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        run_forward=args.run_forward,
        use_real_sample=not args.use_random_input,
        sample=sample,
        official_croma_dir=args.official_croma_dir,
        pretrained_path=args.pretrained_path,
        official_size=args.official_size,
    )

    report["croma_routes"].append(official_route)

    log(
        "INFO",
        f"Official route: import={official_route['import_status']}, "
        f"model={official_route['model_status']}, "
        f"forward={official_route['forward_status']}",
    )

    any_forward_ok = any(
        route["forward_status"] == "ok"
        for route in report["croma_routes"]
    )

    any_model_ok = any(
        route["model_status"] == "ok"
        for route in report["croma_routes"]
    )

    if args.run_forward:
        report["status"] = "ok" if any_forward_ok else "failed"
    else:
        report["status"] = "ok" if any_model_ok else "failed"

    report["recommendations"] = build_recommendations(report)

    log("STEP", "Writing reports.")

    write_json(output_json, report, overwrite=args.overwrite)
    write_markdown(output_md, report, overwrite=args.overwrite)

    log("OK", f"Wrote JSON:     {path_to_str(output_json)}")
    log("OK", f"Wrote Markdown: {path_to_str(output_md)}")

    log("STEP", "Final summary.")
    log("OK", f"Status: {report['status']}")
    log("OK", f"Selected device: {device}")

    for route in report["croma_routes"]:
        log(
            "OK" if route["model_status"] == "ok" else "WARN",
            f"{route['route_name']}: "
            f"import={route['import_status']}, "
            f"model={route['model_status']}, "
            f"forward={route['forward_status']}",
        )

    if report["status"] != "ok":
        log(
            "WARN",
            "CROMA environment is not fully ready yet. Check the Markdown report recommendations.",
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()