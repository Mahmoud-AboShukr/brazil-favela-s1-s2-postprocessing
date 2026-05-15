#!/usr/bin/env python3
"""
Visualize remaining S2/S1 nodata in instance_C_s2_nodata_repaired.

This script is a visual QC companion to:

    05_inspect_instance_c_nodata_and_alignment.py

It creates diagnostic PNGs for each city showing:
- S2 RGB preview
- S2 residual nodata border/internal classification
- S1 VV preview
- S1 residual nodata border/internal classification
- combined S2/S1 nodata mask
- label overlay on S2 RGB

It reads downsampled arrays for visualization, so it is safe for large rasters.
It does not modify any raster.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.enums import Resampling


RGB_BANDS = {
    "red": 4,    # B04
    "green": 3,  # B03
    "blue": 2,   # B02
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize remaining S2/S1 nodata in repaired instance C."
    )

    parser.add_argument(
        "--instance-root",
        type=str,
        required=True,
        help=(
            "Root of instance C, e.g. "
            "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output figure directory. If omitted, figures are written to "
            "<instance-root>/qc/remaining_nodata_figures"
        ),
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional city list. If omitted, all cities under instance C /s2 are visualized.",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=1800,
        help="Maximum width/height for downsampled visualization arrays.",
    )

    parser.add_argument(
        "--border-margin",
        type=int,
        default=128,
        help="Border margin in original pixels for border-connected nodata classification.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output PNG DPI.",
    )

    s2_zero_group = parser.add_mutually_exclusive_group()
    s2_zero_group.add_argument(
        "--s2-all-zero-as-nodata",
        dest="s2_all_zero_as_nodata",
        action="store_true",
        help="Treat S2 all-zero-all-band pixels as nodata.",
    )
    s2_zero_group.add_argument(
        "--no-s2-all-zero-as-nodata",
        dest="s2_all_zero_as_nodata",
        action="store_false",
        help="Do not treat S2 all-zero pixels as nodata.",
    )
    parser.set_defaults(s2_all_zero_as_nodata=True)

    s1_zero_group = parser.add_mutually_exclusive_group()
    s1_zero_group.add_argument(
        "--s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_true",
        help="Treat S1 all-zero-all-band pixels as nodata.",
    )
    s1_zero_group.add_argument(
        "--no-s1-all-zero-as-nodata",
        dest="s1_all_zero_as_nodata",
        action="store_false",
        help="Do not treat S1 all-zero pixels as nodata.",
    )
    parser.set_defaults(s1_all_zero_as_nodata=False)

    parser.add_argument(
        "--nan-as-nodata",
        action="store_true",
        default=True,
        help="Treat NaN/Inf values as nodata.",
    )

    parser.add_argument(
        "--label-positive-threshold",
        type=float,
        default=0.5,
        help="Label pixels > this value are treated as positive.",
    )

    return parser.parse_args()


def compute_output_shape(height: int, width: int, max_size: int) -> tuple[int, int]:
    scale = min(1.0, float(max_size) / float(max(height, width)))
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    return out_h, out_w


def find_s2_raster(instance_root: Path, city: str) -> Path:
    root = instance_root / "s2" / city
    candidates = sorted(root.glob(f"{city}_s2_12bands_reflectance_10m.tif"))
    if not candidates:
        candidates = sorted(root.glob(f"{city}_s2*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No S2 raster found for {city} under {root}")
    return candidates[0]


def find_s1_raster(instance_root: Path, city: str) -> Path:
    root = instance_root / "s1_snap" / city
    candidates = sorted(root.glob(f"{city}_s1_snap_vv_vh_vvdiff_10m_aligned.tif"))
    if not candidates:
        candidates = sorted(root.glob(f"{city}_s1*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No S1 raster found for {city} under {root}")
    return candidates[0]


def find_label_raster(instance_root: Path, city: str) -> Path:
    root = instance_root / "labels" / city
    candidates = sorted(root.glob(f"{city}_label_final.tif"))
    if not candidates:
        candidates = sorted(root.glob(f"{city}_label*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No label raster found for {city} under {root}")
    return candidates[0]


def discover_cities(instance_root: Path, cities: list[str] | None) -> list[str]:
    if cities:
        return sorted(cities)

    s2_root = instance_root / "s2"
    discovered = sorted([p.name for p in s2_root.iterdir() if p.is_dir()])

    if not discovered:
        raise FileNotFoundError(f"No city folders found under {s2_root}")

    return discovered


def percent(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def robust_rgb_stretch(rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)

    for i in range(3):
        band = rgb[i]
        usable = valid_mask & np.isfinite(band)
        values = band[usable]

        if values.size < 10:
            continue

        lo, hi = np.percentile(values, [2, 98])

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))

        if hi <= lo:
            continue

        stretched = np.clip((band - lo) / (hi - lo), 0, 1)
        stretched[~valid_mask] = 0
        out[i] = stretched

    return np.moveaxis(out, 0, -1)


def robust_gray_stretch(gray: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)
    out = np.zeros_like(gray, dtype=np.float32)

    usable = valid_mask & np.isfinite(gray)
    values = gray[usable]

    if values.size < 10:
        return out

    lo, hi = np.percentile(values, [2, 98])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))

    if hi <= lo:
        return out

    out = np.clip((gray - lo) / (hi - lo), 0, 1)
    out[~valid_mask] = 0
    return out


def classify_border_connected(
    nodata_mask: np.ndarray,
    border_margin_rows: int,
    border_margin_cols: int,
) -> np.ndarray:
    nodata_mask = nodata_mask.astype(bool, copy=False)

    if not nodata_mask.any():
        return np.zeros_like(nodata_mask, dtype=bool)

    height, width = nodata_mask.shape
    border_margin_rows = max(1, min(border_margin_rows, height))
    border_margin_cols = max(1, min(border_margin_cols, width))

    seed = np.zeros_like(nodata_mask, dtype=bool)
    seed[:border_margin_rows, :] = True
    seed[-border_margin_rows:, :] = True
    seed[:, :border_margin_cols] = True
    seed[:, -border_margin_cols:] = True
    seed &= nodata_mask

    if not seed.any():
        return np.zeros_like(nodata_mask, dtype=bool)

    try:
        from scipy import ndimage
        return ndimage.binary_propagation(seed, mask=nodata_mask).astype(bool)
    except Exception:
        pass

    visited = np.zeros_like(nodata_mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    rows, cols = np.where(seed)
    for r, c in zip(rows, cols):
        visited[int(r), int(c)] = True
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


def build_downsampled_nodata_mask(
    path: Path,
    out_h: int,
    out_w: int,
    all_zero_as_nodata: bool,
    nan_as_nodata: bool,
) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        band_indexes = list(range(1, src.count + 1))

        masks = src.read_masks(
            indexes=band_indexes,
            out_shape=(src.count, out_h, out_w),
            resampling=Resampling.nearest,
        )

        official = np.any(masks == 0, axis=0)
        combined = official.copy()

        all_zero = np.zeros((out_h, out_w), dtype=bool)
        nonfinite = np.zeros((out_h, out_w), dtype=bool)

        if all_zero_as_nodata or nan_as_nodata:
            data = src.read(
                indexes=band_indexes,
                out_shape=(src.count, out_h, out_w),
                resampling=Resampling.nearest,
            )

            if all_zero_as_nodata:
                all_zero = np.all(data == 0, axis=0)
                combined |= all_zero

            if nan_as_nodata:
                nonfinite = np.any(~np.isfinite(data), axis=0)
                combined |= nonfinite

        meta = {
            "width": src.width,
            "height": src.height,
            "out_width": out_w,
            "out_height": out_h,
            "band_count": src.count,
            "dtype": src.dtypes[0],
            "nodata_value": src.nodata,
            "official_nodata_pixels_downsampled": int(official.sum()),
            "all_zero_pixels_downsampled": int(all_zero.sum()),
            "nonfinite_pixels_downsampled": int(nonfinite.sum()),
            "combined_nodata_pixels_downsampled": int(combined.sum()),
        }

    return combined, meta


def read_s2_rgb(path: Path, out_h: int, out_w: int, valid_mask: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        rgb = src.read(
            indexes=[RGB_BANDS["red"], RGB_BANDS["green"], RGB_BANDS["blue"]],
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
        )
    return robust_rgb_stretch(rgb, valid_mask)


def read_s1_vv(path: Path, out_h: int, out_w: int, valid_mask: np.ndarray) -> np.ndarray:
    with rasterio.open(path) as src:
        vv = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        )
    return robust_gray_stretch(vv, valid_mask)


def read_label_mask(
    path: Path,
    out_h: int,
    out_w: int,
    positive_threshold: float,
) -> np.ndarray:
    with rasterio.open(path) as src:
        label = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.nearest,
        )
    return label > positive_threshold


def make_nodata_class_mask(
    nodata_mask: np.ndarray,
    border_margin_rows: int,
    border_margin_cols: int,
) -> tuple[np.ndarray, dict]:
    border = classify_border_connected(
        nodata_mask=nodata_mask,
        border_margin_rows=border_margin_rows,
        border_margin_cols=border_margin_cols,
    )
    internal = nodata_mask & ~border

    class_mask = np.zeros_like(nodata_mask, dtype=np.uint8)
    class_mask[border] = 1
    class_mask[internal] = 2

    total = nodata_mask.size
    nodata_total = int(nodata_mask.sum())

    stats = {
        "nodata_pixels_downsampled": nodata_total,
        "border_nodata_pixels_downsampled": int(border.sum()),
        "internal_nodata_pixels_downsampled": int(internal.sum()),
        "nodata_percent_downsampled": percent(nodata_total, total),
        "border_percent_downsampled": percent(int(border.sum()), total),
        "internal_percent_downsampled": percent(int(internal.sum()), total),
        "border_share_percent_downsampled": percent(int(border.sum()), nodata_total),
        "internal_share_percent_downsampled": percent(int(internal.sum()), nodata_total),
    }

    return class_mask, stats


def overlay_mask_on_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float],
    alpha: float = 0.55,
) -> np.ndarray:
    out = rgb.copy()
    color_array = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * color_array
    return np.clip(out, 0, 1)


def save_city_figure(
    city: str,
    output_path: Path,
    s2_rgb: np.ndarray,
    s2_class: np.ndarray,
    s1_vv: np.ndarray,
    s1_class: np.ndarray,
    combined_problem: np.ndarray,
    label_mask: np.ndarray,
    s2_stats: dict,
    s1_stats: dict,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].imshow(s2_rgb)
    axes[0, 0].set_title("S2 RGB after crop")
    axes[0, 0].axis("off")

    cmap = ListedColormap(["black", "orange", "red"])
    axes[0, 1].imshow(s2_class, cmap=cmap, vmin=0, vmax=2)
    axes[0, 1].set_title(
        "S2 residual nodata\n"
        f"{s2_stats['nodata_percent_downsampled']:.4f}% total, "
        f"{s2_stats['internal_percent_downsampled']:.4f}% internal"
    )
    axes[0, 1].axis("off")

    axes[0, 2].imshow(s1_vv, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("S1 VV preview after crop")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(s1_class, cmap=cmap, vmin=0, vmax=2)
    axes[1, 0].set_title(
        "S1 residual nodata\n"
        f"{s1_stats['nodata_percent_downsampled']:.4f}% total, "
        f"{s1_stats['internal_percent_downsampled']:.4f}% internal"
    )
    axes[1, 0].axis("off")

    combined_display = np.zeros((*combined_problem.shape, 3), dtype=np.float32)
    # red = S2 only, blue = S1 only, yellow = both
    s2_only = combined_problem == 1
    s1_only = combined_problem == 2
    both = combined_problem == 3
    combined_display[s2_only] = [1, 0, 0]
    combined_display[s1_only] = [0, 0.3, 1]
    combined_display[both] = [1, 1, 0]
    axes[1, 1].imshow(combined_display)
    axes[1, 1].set_title("Combined nodata\nred=S2, blue=S1, yellow=both")
    axes[1, 1].axis("off")

    label_overlay = overlay_mask_on_rgb(
        rgb=s2_rgb,
        mask=label_mask,
        color=(0.0, 1.0, 0.0),
        alpha=0.45,
    )
    axes[1, 2].imshow(label_overlay)
    axes[1, 2].set_title("Label overlay on S2 RGB\ngreen = favela label")
    axes[1, 2].axis("off")

    legend_handles = [
        Patch(facecolor="black", label="valid / non-nodata"),
        Patch(facecolor="orange", label="border-connected nodata"),
        Patch(facecolor="red", label="internal nodata"),
    ]
    axes[0, 1].legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(f"{city} — instance C remaining nodata visual QC", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def inspect_city(
    city: str,
    instance_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    s2_path = find_s2_raster(instance_root, city)
    s1_path = find_s1_raster(instance_root, city)
    label_path = find_label_raster(instance_root, city)

    with rasterio.open(s2_path) as src:
        out_h, out_w = compute_output_shape(src.height, src.width, args.max_size)
        scale_y = src.height / out_h
        scale_x = src.width / out_w

    border_margin_rows = max(1, int(round(args.border_margin / scale_y)))
    border_margin_cols = max(1, int(round(args.border_margin / scale_x)))

    s2_nodata, s2_meta = build_downsampled_nodata_mask(
        path=s2_path,
        out_h=out_h,
        out_w=out_w,
        all_zero_as_nodata=args.s2_all_zero_as_nodata,
        nan_as_nodata=args.nan_as_nodata,
    )

    s1_nodata, s1_meta = build_downsampled_nodata_mask(
        path=s1_path,
        out_h=out_h,
        out_w=out_w,
        all_zero_as_nodata=args.s1_all_zero_as_nodata,
        nan_as_nodata=args.nan_as_nodata,
    )

    s2_class, s2_stats = make_nodata_class_mask(
        nodata_mask=s2_nodata,
        border_margin_rows=border_margin_rows,
        border_margin_cols=border_margin_cols,
    )

    s1_class, s1_stats = make_nodata_class_mask(
        nodata_mask=s1_nodata,
        border_margin_rows=border_margin_rows,
        border_margin_cols=border_margin_cols,
    )

    s2_rgb = read_s2_rgb(
        path=s2_path,
        out_h=out_h,
        out_w=out_w,
        valid_mask=~s2_nodata,
    )

    s1_vv = read_s1_vv(
        path=s1_path,
        out_h=out_h,
        out_w=out_w,
        valid_mask=~s1_nodata,
    )

    label_mask = read_label_mask(
        path=label_path,
        out_h=out_h,
        out_w=out_w,
        positive_threshold=args.label_positive_threshold,
    )

    combined_problem = np.zeros_like(s2_nodata, dtype=np.uint8)
    combined_problem[s2_nodata] += 1
    combined_problem[s1_nodata] += 2

    figure_path = output_dir / f"{city}_instance_c_remaining_nodata.png"

    save_city_figure(
        city=city,
        output_path=figure_path,
        s2_rgb=s2_rgb,
        s2_class=s2_class,
        s1_vv=s1_vv,
        s1_class=s1_class,
        combined_problem=combined_problem,
        label_mask=label_mask,
        s2_stats=s2_stats,
        s1_stats=s1_stats,
        dpi=args.dpi,
    )

    total_pixels = int(s2_nodata.size)
    label_positive = int(label_mask.sum())
    s2_label_overlap = int((s2_nodata & label_mask).sum())
    s1_label_overlap = int((s1_nodata & label_mask).sum())

    row = {
        "city": city,
        "figure_path": str(figure_path),
        "s2_path": str(s2_path),
        "s1_path": str(s1_path),
        "label_path": str(label_path),
        "original_width": s2_meta["width"],
        "original_height": s2_meta["height"],
        "render_width": out_w,
        "render_height": out_h,
        "s2_nodata_percent_downsampled": s2_stats["nodata_percent_downsampled"],
        "s2_border_percent_downsampled": s2_stats["border_percent_downsampled"],
        "s2_internal_percent_downsampled": s2_stats["internal_percent_downsampled"],
        "s2_border_share_percent_downsampled": s2_stats["border_share_percent_downsampled"],
        "s2_internal_share_percent_downsampled": s2_stats["internal_share_percent_downsampled"],
        "s1_nodata_percent_downsampled": s1_stats["nodata_percent_downsampled"],
        "s1_border_percent_downsampled": s1_stats["border_percent_downsampled"],
        "s1_internal_percent_downsampled": s1_stats["internal_percent_downsampled"],
        "s1_border_share_percent_downsampled": s1_stats["border_share_percent_downsampled"],
        "s1_internal_share_percent_downsampled": s1_stats["internal_share_percent_downsampled"],
        "label_positive_pixels_downsampled": label_positive,
        "label_positive_percent_downsampled": percent(label_positive, total_pixels),
        "s2_nodata_label_overlap_pixels_downsampled": s2_label_overlap,
        "s2_nodata_label_overlap_percent_of_label_downsampled": percent(s2_label_overlap, label_positive),
        "s1_nodata_label_overlap_pixels_downsampled": s1_label_overlap,
        "s1_nodata_label_overlap_percent_of_label_downsampled": percent(s1_label_overlap, label_positive),
        "border_margin_original_pixels": args.border_margin,
        "border_margin_render_rows": border_margin_rows,
        "border_margin_render_cols": border_margin_cols,
    }

    return row


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    fields = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    instance_root = Path(args.instance_root)
    if not instance_root.exists():
        raise FileNotFoundError(f"Instance root does not exist: {instance_root}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else instance_root / "qc" / "remaining_nodata_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cities = discover_cities(instance_root, args.cities)

    print(f"[INFO] Instance root: {instance_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Cities to visualize: {len(cities)}")
    print(f"[INFO] max_size: {args.max_size}")
    print(f"[INFO] border_margin: {args.border_margin}")
    print(f"[INFO] S2 all-zero-as-nodata: {args.s2_all_zero_as_nodata}")
    print(f"[INFO] S1 all-zero-as-nodata: {args.s1_all_zero_as_nodata}")

    rows = []

    for i, city in enumerate(cities, start=1):
        print(f"\n[STEP {i}/{len(cities)}] {city}")

        try:
            row = inspect_city(
                city=city,
                instance_root=instance_root,
                output_dir=output_dir,
                args=args,
            )
            rows.append(row)

            print(
                "[OK] "
                f"S2 nodata={row['s2_nodata_percent_downsampled']:.4f}% | "
                f"S1 nodata={row['s1_nodata_percent_downsampled']:.4f}% | "
                f"S2-label overlap={row['s2_nodata_label_overlap_percent_of_label_downsampled']:.4f}% | "
                f"S1-label overlap={row['s1_nodata_label_overlap_percent_of_label_downsampled']:.4f}%"
            )

        except Exception as exc:
            print(f"[ERROR] {city}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "city": city,
                    "figure_path": "",
                    "error": str(exc),
                }
            )

    summary_path = output_dir / "instance_C_remaining_nodata_visual_summary.csv"
    write_csv(rows, summary_path)

    print(f"\n[DONE] Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()