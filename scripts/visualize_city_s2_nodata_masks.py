#!/usr/bin/env python3
"""
Visualize city-level Sentinel-2 nodata masks.

This script inspects standardized 12-band Sentinel-2 city GeoTIFFs and creates
diagnostic PNG figures showing:

1. RGB preview from B04/B03/B02.
2. Combined nodata mask.
3. Nodata overlay on RGB.
4. Border-connected vs internal nodata classification.

The script is designed for large rasters. It reads downsampled arrays for
visualization instead of loading full-resolution city rasters into memory.
Exact city-level nodata percentages should still be taken from
inspect_city_s2_composite_nodata.py; this script is primarily visual QC.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling


S2_BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
]

# 1-based raster band indices for RGB using the expected band order:
# B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
RGB_BANDS = {
    "red": 4,    # B04
    "green": 3,  # B03
    "blue": 2,   # B02
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize nodata masks in city-level standardized Sentinel-2 rasters."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional config path. Currently used only as fallback for output root "
            "if --s2-root is not provided."
        ),
    )

    parser.add_argument(
        "--s2-root",
        type=str,
        default=None,
        help=(
            "Root folder containing city S2 rasters, e.g. "
            "D:/post_processing_dataset/dataset_instances/instance_B_standard_rs/s2"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where PNG figures and summary CSV will be written.",
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help=(
            "Optional list of city slugs to inspect. "
            "Example: --cities sao_paulo belo_horizonte brasilia"
        ),
    )

    zero_group = parser.add_mutually_exclusive_group()
    zero_group.add_argument(
        "--all-zero-as-nodata",
        dest="all_zero_as_nodata",
        action="store_true",
        help="Treat pixels where all 12 bands are exactly zero as nodata.",
    )
    zero_group.add_argument(
        "--no-all-zero-as-nodata",
        dest="all_zero_as_nodata",
        action="store_false",
        help="Do not treat all-zero-all-band pixels as nodata.",
    )
    parser.set_defaults(all_zero_as_nodata=True)

    parser.add_argument(
        "--border-margin",
        type=int,
        default=128,
        help=(
            "Border margin in original raster pixels. Nodata connected to this "
            "border seed zone is classified as border-connected nodata."
        ),
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=1800,
        help=(
            "Maximum width or height of downsampled visualization arrays. "
            "Increase for more detail, decrease for lower memory use."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output PNG DPI.",
    )

    return parser.parse_args()


def load_config_yaml(config_path: str | None) -> dict:
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        print(f"[WARN] Config file does not exist: {path}", file=sys.stderr)
        return {}

    try:
        import yaml
    except ImportError:
        print(
            "[WARN] PyYAML is not installed, so --config cannot be parsed. "
            "Pass --s2-root explicitly.",
            file=sys.stderr,
        )
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        return {}

    return data


def infer_s2_root_from_config(config: dict) -> Path | None:
    """
    Best-effort fallback only.

    Because configs/default.yaml may differ between machines, this function tries
    a few likely keys but does not assume one fixed schema.
    """
    candidate_roots = []

    for key in ["output_root", "post_processing_root", "postprocessing_root"]:
        value = config.get(key)
        if value:
            candidate_roots.append(Path(value))

    paths = config.get("paths", {})
    if isinstance(paths, dict):
        for key in ["output_root", "post_processing_root", "postprocessing_root"]:
            value = paths.get(key)
            if value:
                candidate_roots.append(Path(value))

    for root in candidate_roots:
        candidate = root / "dataset_instances" / "instance_B_standard_rs" / "s2"
        if candidate.exists():
            return candidate

    return None


def resolve_s2_root(args: argparse.Namespace) -> Path:
    if args.s2_root is not None:
        root = Path(args.s2_root)
        if not root.exists():
            raise FileNotFoundError(f"--s2-root does not exist: {root}")
        return root

    config = load_config_yaml(args.config)
    inferred = infer_s2_root_from_config(config)

    if inferred is not None:
        return inferred

    raise ValueError(
        "Could not resolve S2 root. Please pass --s2-root explicitly. "
        "This is recommended because configs/default.yaml may contain "
        "machine-specific Linux or Windows paths."
    )


def find_city_raster(s2_root: Path, city: str) -> Path:
    city_dir = s2_root / city

    patterns = [
        f"{city}_s2_12bands_reflectance_10m.tif",
        f"{city}_s2*.tif",
        "*_s2_12bands_reflectance_10m.tif",
        "*.tif",
    ]

    candidates: list[Path] = []

    if city_dir.exists():
        for pattern in patterns:
            candidates.extend(sorted(city_dir.glob(pattern)))

    if not candidates:
        for pattern in patterns:
            candidates.extend(sorted(s2_root.glob(pattern)))
            candidates.extend(sorted(s2_root.glob(f"*/{pattern}")))

    candidates = [p for p in candidates if p.is_file() and city in p.name]

    if not candidates:
        raise FileNotFoundError(f"No S2 raster found for city '{city}' under {s2_root}")

    if len(candidates) > 1:
        exact = [p for p in candidates if p.name == f"{city}_s2_12bands_reflectance_10m.tif"]
        if exact:
            return exact[0]

    return candidates[0]


def discover_city_rasters(s2_root: Path, cities: Iterable[str] | None) -> dict[str, Path]:
    if cities:
        return {city: find_city_raster(s2_root, city) for city in cities}

    discovered: dict[str, Path] = {}

    for path in sorted(s2_root.glob("*/*_s2_12bands_reflectance_10m.tif")):
        city = path.parent.name
        discovered[city] = path

    if not discovered:
        for path in sorted(s2_root.glob("*_s2_12bands_reflectance_10m.tif")):
            city = path.name.replace("_s2_12bands_reflectance_10m.tif", "")
            discovered[city] = path

    if not discovered:
        raise FileNotFoundError(f"No S2 rasters discovered under {s2_root}")

    return discovered


def compute_output_shape(height: int, width: int, max_size: int) -> tuple[int, int]:
    if max_size <= 0:
        raise ValueError("--max-size must be positive")

    scale = min(1.0, max_size / max(height, width))
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    return out_h, out_w


def robust_rgb_stretch(rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Convert a 3 x H x W RGB array into display-ready H x W x 3 float image.

    Stretch each channel independently using robust percentiles over valid pixels.
    """
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)

    for i in range(3):
        band = rgb[i]
        finite_valid = valid_mask & np.isfinite(band)
        values = band[finite_valid]

        if values.size < 10:
            continue

        lo, hi = np.percentile(values, [2, 98])

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))

        if hi <= lo:
            continue

        stretched = (band - lo) / (hi - lo)
        stretched = np.clip(stretched, 0, 1)
        stretched[~valid_mask] = 0
        out[i] = stretched

    return np.moveaxis(out, 0, -1)


def flood_fill_border_connected(
    nodata_mask: np.ndarray,
    border_margin_rows: int,
    border_margin_cols: int,
) -> np.ndarray:
    """
    Classify nodata connected to a border seed zone.

    The seed zone is not only the outermost image edge; it includes a configurable
    margin, scaled from original raster pixels to visualization pixels.
    """
    nodata_mask = nodata_mask.astype(bool)
    height, width = nodata_mask.shape

    border_margin_rows = max(1, min(border_margin_rows, height))
    border_margin_cols = max(1, min(border_margin_cols, width))

    seed = np.zeros_like(nodata_mask, dtype=bool)
    seed[:border_margin_rows, :] = True
    seed[-border_margin_rows:, :] = True
    seed[:, :border_margin_cols] = True
    seed[:, -border_margin_cols:] = True
    seed &= nodata_mask

    # Use scipy if available because it is faster. Fall back to pure Python BFS.
    try:
        from scipy import ndimage

        return ndimage.binary_propagation(seed, mask=nodata_mask).astype(bool)
    except Exception:
        pass

    visited = np.zeros_like(nodata_mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    seed_rows, seed_cols = np.where(seed)
    for r, c in zip(seed_rows, seed_cols):
        visited[r, c] = True
        q.append((int(r), int(c)))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        r, c = q.popleft()

        for dr, dc in neighbors:
            rr = r + dr
            cc = c + dc

            if rr < 0 or rr >= height or cc < 0 or cc >= width:
                continue

            if visited[rr, cc] or not nodata_mask[rr, cc]:
                continue

            visited[rr, cc] = True
            q.append((rr, cc))

    return visited


def make_overlay(rgb: np.ndarray, nodata_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = rgb.copy()
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    overlay[nodata_mask] = (1 - alpha) * overlay[nodata_mask] + alpha * red
    return np.clip(overlay, 0, 1)


def read_downsampled_city(path: Path, max_size: int, all_zero_as_nodata: bool) -> dict:
    with rasterio.open(path) as src:
        if src.count < 4:
            raise ValueError(f"Expected at least 4 S2 bands, found {src.count}: {path}")

        out_h, out_w = compute_output_shape(src.height, src.width, max_size)

        masks = src.read_masks(
            indexes=list(range(1, src.count + 1)),
            out_shape=(src.count, out_h, out_w),
            resampling=Resampling.nearest,
        )

        official_nodata = np.any(masks == 0, axis=0)

        all_zero_nodata = np.zeros((out_h, out_w), dtype=bool)

        if all_zero_as_nodata:
            data_all = src.read(
                indexes=list(range(1, src.count + 1)),
                out_shape=(src.count, out_h, out_w),
                resampling=Resampling.nearest,
            )
            all_zero_nodata = np.all(data_all == 0, axis=0)

        combined_nodata = official_nodata | all_zero_nodata

        rgb = src.read(
            indexes=[RGB_BANDS["red"], RGB_BANDS["green"], RGB_BANDS["blue"]],
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
        )

        valid_for_rgb = ~combined_nodata
        rgb_display = robust_rgb_stretch(rgb, valid_for_rgb)

        scale_y = src.height / out_h
        scale_x = src.width / out_w

        return {
            "rgb_display": rgb_display,
            "official_nodata": official_nodata,
            "all_zero_nodata": all_zero_nodata,
            "combined_nodata": combined_nodata,
            "original_height": src.height,
            "original_width": src.width,
            "render_height": out_h,
            "render_width": out_w,
            "scale_y": scale_y,
            "scale_x": scale_x,
            "crs": str(src.crs),
            "transform": src.transform,
            "nodata": src.nodata,
            "count": src.count,
            "dtype": src.dtypes[0],
        }


def percentage(mask: np.ndarray) -> float:
    return 100.0 * float(mask.sum()) / float(mask.size)


def save_figure(
    city: str,
    raster_path: Path,
    output_path: Path,
    data: dict,
    border_margin: int,
    dpi: int,
) -> dict:
    rgb = data["rgb_display"]
    official_nodata = data["official_nodata"]
    all_zero_nodata = data["all_zero_nodata"]
    combined_nodata = data["combined_nodata"]

    margin_rows = max(1, int(round(border_margin / data["scale_y"])))
    margin_cols = max(1, int(round(border_margin / data["scale_x"])))

    border_connected = flood_fill_border_connected(
        combined_nodata,
        border_margin_rows=margin_rows,
        border_margin_cols=margin_cols,
    )
    internal_nodata = combined_nodata & ~border_connected

    class_mask = np.zeros_like(combined_nodata, dtype=np.uint8)
    class_mask[border_connected] = 1
    class_mask[internal_nodata] = 2

    overlay = make_overlay(rgb, combined_nodata)

    combined_pct = percentage(combined_nodata)
    official_pct = percentage(official_nodata)
    all_zero_pct = percentage(all_zero_nodata)
    border_pct = percentage(border_connected)
    internal_pct = percentage(internal_nodata)

    nodata_total = int(combined_nodata.sum())
    border_share = 0.0 if nodata_total == 0 else 100.0 * float(border_connected.sum()) / nodata_total
    internal_share = 0.0 if nodata_total == 0 else 100.0 * float(internal_nodata.sum()) / nodata_total

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("S2 RGB preview: B04/B03/B02")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(combined_nodata, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Combined nodata mask\n{combined_pct:.3f}% of visualization pixels")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Nodata overlay on RGB")
    axes[1, 0].axis("off")

    cmap = ListedColormap(["black", "orange", "red"])
    axes[1, 1].imshow(class_mask, cmap=cmap, vmin=0, vmax=2)
    axes[1, 1].set_title(
        "Border-connected vs internal nodata\n"
        f"border share={border_share:.2f}%, internal share={internal_share:.2f}%"
    )
    axes[1, 1].axis("off")

    legend_handles = [
        Patch(facecolor="black", label="valid / non-nodata"),
        Patch(facecolor="orange", label="border-connected nodata"),
        Patch(facecolor="red", label="internal nodata"),
    ]
    axes[1, 1].legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=1,
        fontsize=9,
        frameon=True,
    )

    fig.suptitle(
        f"{city}\n"
        f"{raster_path}\n"
        f"Original: {data['original_width']} x {data['original_height']} | "
        f"Rendered: {data['render_width']} x {data['render_height']} | "
        f"Bands: {data['count']} | dtype: {data['dtype']} | nodata: {data['nodata']}",
        fontsize=11,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "city": city,
        "raster_path": str(raster_path),
        "figure_path": str(output_path),
        "original_width": data["original_width"],
        "original_height": data["original_height"],
        "render_width": data["render_width"],
        "render_height": data["render_height"],
        "band_count": data["count"],
        "dtype": data["dtype"],
        "raster_nodata_value": data["nodata"],
        "crs": data["crs"],
        "official_nodata_percent_downsampled": official_pct,
        "all_zero_allbands_percent_downsampled": all_zero_pct,
        "combined_nodata_percent_downsampled": combined_pct,
        "border_connected_nodata_percent_downsampled": border_pct,
        "internal_nodata_percent_downsampled": internal_pct,
        "border_share_of_nodata_percent_downsampled": border_share,
        "internal_share_of_nodata_percent_downsampled": internal_share,
        "border_margin_original_pixels": border_margin,
        "border_margin_render_rows": margin_rows,
        "border_margin_render_cols": margin_cols,
    }


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    s2_root = resolve_s2_root(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    city_rasters = discover_city_rasters(s2_root, args.cities)

    print(f"[INFO] S2 root: {s2_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Cities to process: {len(city_rasters)}")
    print(f"[INFO] all_zero_as_nodata: {args.all_zero_as_nodata}")
    print(f"[INFO] border_margin: {args.border_margin}")
    print(f"[INFO] max_size: {args.max_size}")

    summary_rows: list[dict] = []

    for city, raster_path in city_rasters.items():
        print(f"\n[STEP] {city}")
        print(f"[INFO] Raster: {raster_path}")

        try:
            data = read_downsampled_city(
                raster_path,
                max_size=args.max_size,
                all_zero_as_nodata=args.all_zero_as_nodata,
            )

            figure_path = output_dir / f"{city}_s2_nodata_diagnostic.png"

            row = save_figure(
                city=city,
                raster_path=raster_path,
                output_path=figure_path,
                data=data,
                border_margin=args.border_margin,
                dpi=args.dpi,
            )

            summary_rows.append(row)

            print(f"[OK] Saved: {figure_path}")
            print(
                "[INFO] Downsampled combined nodata: "
                f"{row['combined_nodata_percent_downsampled']:.4f}% | "
                f"border share: {row['border_share_of_nodata_percent_downsampled']:.2f}% | "
                f"internal share: {row['internal_share_of_nodata_percent_downsampled']:.2f}%"
            )

        except Exception as exc:
            print(f"[ERROR] Failed city {city}: {exc}", file=sys.stderr)
            summary_rows.append(
                {
                    "city": city,
                    "raster_path": str(raster_path),
                    "figure_path": "",
                    "error": str(exc),
                }
            )

    summary_path = output_dir / "city_s2_nodata_visual_summary.csv"
    write_summary_csv(summary_rows, summary_path)

    print(f"\n[DONE] Summary CSV: {summary_path}")
    print(
        "[NOTE] Percentages in this CSV are computed on downsampled visualization arrays. "
        "Use inspect_city_s2_composite_nodata.py for exact full-resolution nodata statistics."
    )


if __name__ == "__main__":
    main()