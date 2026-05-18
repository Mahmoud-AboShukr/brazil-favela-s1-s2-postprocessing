#!/usr/bin/env python3
"""
Inspect nodata in city-level Sentinel-2 composite rasters.

Purpose
-------
This script checks the S2 city-level GeoTIFFs before patch tiling. It reports,
for each city:

    - total pixels
    - official/masked nodata percentage
    - all-zero-all-bands percentage
    - combined nodata percentage
    - whether nodata is mostly near borders or inside the city raster

This helps diagnose whether S2 nodata comes from:
    - edge/coverage effects
    - cloud-removal/compositing holes
    - invalid all-zero composite pixels

Default input
-------------
By default, the script scans:

    <output_root>/dataset_instances/instance_B_standard_rs/s2

Example
-------
python scripts/inspect_city_s2_composite_nodata.py \
  --config configs/default.yaml

If you want to inspect another S2 composite folder:

python scripts/inspect_city_s2_composite_nodata.py \
  --config configs/default.yaml \
  --s2-root /path/to/s2/composites
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import yaml

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect nodata in city-level S2 composite rasters."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--s2-root",
        type=Path,
        default=None,
        help=(
            "Root folder containing city-level S2 GeoTIFFs. "
            "If omitted, uses <output_root>/dataset_instances/instance_B_standard_rs/s2."
        ),
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.tif",
        help="Glob pattern for S2 GeoTIFFs under --s2-root. Default: *.tif",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search recursively under S2 root. Default: True.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive search.",
    )
    parser.add_argument(
        "--all-zero-as-nodata",
        action="store_true",
        default=True,
        help="Treat pixels where all S2 bands are exactly zero as nodata. Default: True.",
    )
    parser.add_argument(
        "--no-all-zero-as-nodata",
        dest="all_zero_as_nodata",
        action="store_false",
        help="Do not treat all-zero pixels as nodata.",
    )
    parser.add_argument(
        "--border-margin",
        type=int,
        default=128,
        help=(
            "Pixel margin used to separate border nodata from internal nodata. "
            "Default: 128 pixels."
        ),
    )
    parser.add_argument(
        "--significant-threshold-percent",
        type=float,
        default=1.0,
        help="Threshold to mark a city as having significant nodata. Default: 1 percent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. If omitted, uses <output_root>/qc/city_s2_nodata.",
    )

    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "output_root" not in cfg:
        raise KeyError("Missing output_root in config.")

    return cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_s2_files(root: Path, pattern: str, recursive: bool) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"S2 root not found: {root}")

    if recursive:
        files = sorted(root.rglob(pattern))
    else:
        files = sorted(root.glob(pattern))

    files = [p for p in files if p.is_file()]

    if not files:
        raise FileNotFoundError(f"No S2 files found under {root} with pattern {pattern}")

    return files


def infer_city_name(path: Path) -> str:
    """
    Expected structure:
        .../s2/<city>/<city>_s2_12bands_reflectance_10m.tif

    If this structure is not available, fall back to filename.
    """
    parent = path.parent.name

    if parent and parent.lower() not in {"s2", "rasters", "images"}:
        return parent

    name = path.stem
    for token in [
        "_s2_12bands_reflectance_10m",
        "_s2_reflectance_10m",
        "_s2",
    ]:
        if token in name:
            return name.split(token)[0]

    return name


def iter_windows(src: rasterio.io.DatasetReader) -> List[Window]:
    """
    Use internal block windows if available. This keeps memory safe.
    """
    windows = []

    try:
        for _, window in src.block_windows(1):
            windows.append(window)
    except Exception:
        pass

    if windows:
        return windows

    # Fallback: manual 1024 x 1024 windows.
    tile = 1024

    for row_off in range(0, src.height, tile):
        for col_off in range(0, src.width, tile):
            height = min(tile, src.height - row_off)
            width = min(tile, src.width - col_off)
            windows.append(
                Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=width,
                    height=height,
                )
            )

    return windows


def border_mask_for_window(
    window: Window,
    raster_height: int,
    raster_width: int,
    margin: int,
) -> np.ndarray:
    row_start = int(window.row_off)
    col_start = int(window.col_off)
    height = int(window.height)
    width = int(window.width)

    rows = np.arange(row_start, row_start + height)
    cols = np.arange(col_start, col_start + width)

    row_border = (rows < margin) | (rows >= raster_height - margin)
    col_border = (cols < margin) | (cols >= raster_width - margin)

    return row_border[:, None] | col_border[None, :]


def inspect_one_s2(
    path: Path,
    border_margin: int,
    all_zero_as_nodata: bool,
) -> Dict[str, Any]:
    city = infer_city_name(path)

    with rasterio.open(path) as src:
        width = int(src.width)
        height = int(src.height)
        count = int(src.count)
        crs = str(src.crs)
        transform = src.transform
        nodata_value = src.nodata
        dtype_list = list(src.dtypes)

        total_pixels = 0

        official_or_masked_pixels = 0
        all_zero_pixels = 0
        combined_nodata_pixels = 0

        border_pixels = 0
        internal_pixels = 0
        border_nodata_pixels = 0
        internal_nodata_pixels = 0

        band_nodata_pixels = np.zeros(count, dtype=np.int64)
        band_zero_pixels = np.zeros(count, dtype=np.int64)

        windows = iter_windows(src)

        for window in windows:
            arr = src.read(window=window, masked=True)

            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]

            arr_float = arr.astype("float32")
            data = arr_float.filled(np.nan)

            mask = np.ma.getmaskarray(arr_float)
            if mask.ndim == 3:
                official_pixel = mask.any(axis=0)
                band_mask = mask
            else:
                official_pixel = mask
                band_mask = mask[np.newaxis, :, :]

            nonfinite_pixel = ~np.isfinite(data).all(axis=0)
            official_or_masked = official_pixel | nonfinite_pixel

            finite_all_bands = np.isfinite(data).all(axis=0)
            all_zero = finite_all_bands & np.all(data == 0.0, axis=0)

            if all_zero_as_nodata:
                combined = official_or_masked | all_zero
            else:
                combined = official_or_masked

            n_pixels = int(combined.size)
            total_pixels += n_pixels

            official_or_masked_pixels += int(official_or_masked.sum())
            all_zero_pixels += int(all_zero.sum())
            combined_nodata_pixels += int(combined.sum())

            border_region = border_mask_for_window(
                window=window,
                raster_height=height,
                raster_width=width,
                margin=border_margin,
            )
            internal_region = ~border_region

            border_pixels += int(border_region.sum())
            internal_pixels += int(internal_region.sum())

            border_nodata_pixels += int((combined & border_region).sum())
            internal_nodata_pixels += int((combined & internal_region).sum())

            band_nodata_pixels += band_mask.reshape(count, -1).sum(axis=1).astype(np.int64)

            finite = np.isfinite(data)
            zeros = finite & (data == 0.0)
            band_zero_pixels += zeros.reshape(count, -1).sum(axis=1).astype(np.int64)

    def pct(value: float, denom: float) -> float:
        return 100.0 * float(value) / max(float(denom), 1.0)

    combined_pct = pct(combined_nodata_pixels, total_pixels)

    row: Dict[str, Any] = {
        "city": city,
        "path": str(path),
        "width": width,
        "height": height,
        "bands": count,
        "crs": crs,
        "dtype": ",".join(dtype_list),
        "nodata_value": nodata_value,
        "total_pixels": total_pixels,
        "official_or_masked_nodata_pixels": official_or_masked_pixels,
        "official_or_masked_nodata_percent": pct(official_or_masked_pixels, total_pixels),
        "all_zero_allbands_pixels": all_zero_pixels,
        "all_zero_allbands_percent": pct(all_zero_pixels, total_pixels),
        "combined_nodata_pixels": combined_nodata_pixels,
        "combined_nodata_percent": combined_pct,
        "border_margin_pixels": border_margin,
        "border_pixels": border_pixels,
        "internal_pixels": internal_pixels,
        "border_nodata_pixels": border_nodata_pixels,
        "internal_nodata_pixels": internal_nodata_pixels,
        "border_nodata_percent_of_border": pct(border_nodata_pixels, border_pixels),
        "internal_nodata_percent_of_internal": pct(internal_nodata_pixels, internal_pixels),
        "border_share_of_all_nodata_percent": pct(border_nodata_pixels, combined_nodata_pixels),
        "internal_share_of_all_nodata_percent": pct(internal_nodata_pixels, combined_nodata_pixels),
    }

    for i in range(count):
        band_id = i + 1
        row[f"band_{band_id:02d}_masked_nodata_percent"] = pct(
            band_nodata_pixels[i],
            total_pixels,
        )
        row[f"band_{band_id:02d}_zero_percent"] = pct(
            band_zero_pixels[i],
            total_pixels,
        )

    return row


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))

    s2_root = (
        args.s2_root
        if args.s2_root is not None
        else output_root / "dataset_instances" / "instance_B_standard_rs" / "s2"
    )

    out_dir = args.output_dir or output_root / "qc" / "city_s2_nodata"
    ensure_dir(out_dir)

    files = find_s2_files(
        root=s2_root,
        pattern=args.glob,
        recursive=args.recursive,
    )

    print("[INFO] Inspect city-level S2 composite nodata")
    print(f"[INFO] S2 root: {s2_root}")
    print(f"[INFO] Files found: {len(files)}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] All-zero-all-bands as nodata: {args.all_zero_as_nodata}")
    print(f"[INFO] Border margin: {args.border_margin} pixels")
    print(f"[INFO] Significant threshold: {args.significant_threshold_percent:.3f}%")

    iterator = files
    if tqdm is not None:
        iterator = tqdm(files, desc="Inspecting S2 city rasters")

    rows: List[Dict[str, Any]] = []

    for path in iterator:
        rows.append(
            inspect_one_s2(
                path=path,
                border_margin=args.border_margin,
                all_zero_as_nodata=args.all_zero_as_nodata,
            )
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("combined_nodata_percent", ascending=False).reset_index(drop=True)

    threshold = args.significant_threshold_percent
    df["has_any_combined_nodata"] = df["combined_nodata_pixels"] > 0
    df["has_significant_combined_nodata"] = df["combined_nodata_percent"] >= threshold

    out_csv = out_dir / "city_s2_composite_nodata_summary.csv"
    out_md = out_dir / "city_s2_composite_nodata_summary.md"

    df.to_csv(out_csv, index=False)

    display_cols = [
        "city",
        "width",
        "height",
        "bands",
        "combined_nodata_percent",
        "official_or_masked_nodata_percent",
        "all_zero_allbands_percent",
        "border_share_of_all_nodata_percent",
        "internal_share_of_all_nodata_percent",
        "has_significant_combined_nodata",
        "path",
    ]

    display_df = df[display_cols].copy()

    md_lines = []
    md_lines.append("# City-level S2 Composite Nodata Summary")
    md_lines.append("")
    md_lines.append(f"S2 root: `{s2_root}`")
    md_lines.append("")
    md_lines.append(f"Number of city rasters: **{len(df)}**")
    md_lines.append("")
    md_lines.append(
        f"Cities with any combined nodata: **{int(df['has_any_combined_nodata'].sum())} / {len(df)}**"
    )
    md_lines.append(
        f"Cities with combined nodata >= {threshold:.2f}%: "
        f"**{int(df['has_significant_combined_nodata'].sum())} / {len(df)}**"
    )
    md_lines.append("")
    md_lines.append("## Sorted city table")
    md_lines.append("")
    md_lines.append(display_df.to_markdown(index=False))

    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[INFO] Wrote CSV: {out_csv}")
    print(f"[INFO] Wrote Markdown: {out_md}")

    print("\n[INFO] Top cities by combined S2 nodata percent:")
    print(
        df[
            [
                "city",
                "combined_nodata_percent",
                "official_or_masked_nodata_percent",
                "all_zero_allbands_percent",
                "border_share_of_all_nodata_percent",
                "internal_share_of_all_nodata_percent",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\n[INFO] Global city-level summary:")
    print(f"       cities total: {len(df)}")
    print(f"       cities with any S2 nodata: {int(df['has_any_combined_nodata'].sum())}")
    print(
        f"       cities with >= {threshold:.2f}% S2 nodata: "
        f"{int(df['has_significant_combined_nodata'].sum())}"
    )
    print(f"       mean combined S2 nodata %: {df['combined_nodata_percent'].mean():.4f}")
    print(f"       median combined S2 nodata %: {df['combined_nodata_percent'].median():.4f}")
    print(f"       max combined S2 nodata %: {df['combined_nodata_percent'].max():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())