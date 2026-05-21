#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
12_summarize_s1_statistics_paper_quality_224.py

Paper-quality summary for Option B patch-based Sentinel-1 statistics.

This script does NOT reread rasters. It summarizes the outputs from:

    11_compute_s1_snap_rtc_statistics_224.py

Expected inputs:

    <instance-root>/metadata/croma_probing/s1_statistics_patch_based/
        s1_patch_based_scale_detection_ps224_st112_cover.csv
        s1_patch_based_global_statistics_ps224_st112_cover.csv
        s1_patch_based_city_statistics_ps224_st112_cover.csv
        s1_patch_based_ssl4eo_comparison_ps224_st112_cover.csv
        s1_patch_based_outlier_diagnostics_ps224_st112_cover.csv

Outputs:

    <instance-root>/metadata/croma_probing/s1_statistics_paper_quality/
        paper_s1_statistics_main_table_ps224_st112_cover.csv
        paper_s1_statistics_snap_vs_rtc_table_ps224_st112_cover.csv
        paper_s1_statistics_city_extremes_ps224_st112_cover.csv
        paper_s1_statistics_latex_table_ps224_st112_cover.tex
        paper_s1_statistics_summary_ps224_st112_cover.json
        paper_s1_statistics_summary_ps224_st112_cover.md

Optional figures:

    figures/paper_s1_mean_std_comparison_ps224_st112_cover.png
    figures/paper_s1_outlier_comparison_ps224_st112_cover.png
    figures/paper_s1_city_mean_extremes_vv_ps224_st112_cover.png
    figures/paper_s1_city_mean_extremes_vh_ps224_st112_cover.png

Example:

python src/croma_probing/12_summarize_s1_statistics_paper_quality_224.py `
  --instance-root "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired" `
  --patch-size 224 `
  --stride 112 `
  --edge-mode cover `
  --make-figures `
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


SSL4EO_REFERENCE = {
    "VV": {"mean": -12.59, "std": 5.26},
    "VH": {"mean": -20.26, "std": 5.91},
}


# ---------------------------------------------------------------------
# Logging and utilities
# ---------------------------------------------------------------------

def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def fail(message: str, exit_code: int = 1) -> None:
    log("ERROR", message)
    raise SystemExit(exit_code)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def round_float(value: float, digits: int = 4) -> float:
    return round(safe_float(value), digits)


def fmt(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


# ---------------------------------------------------------------------
# CSV / JSON / Markdown / LaTeX
# ---------------------------------------------------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        fail(f"Required input CSV does not exist: {path_to_str(path)}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        fail(f"Input CSV is empty: {path_to_str(path)}")

    return rows


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


def write_text(path: Path, text: str, overwrite: bool) -> None:
    ensure_output_can_be_written(path, overwrite)

    with path.open("w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------

def stats_dir(instance_root: Path) -> Path:
    return instance_root / "metadata" / "croma_probing" / "s1_statistics_patch_based"


def scale_path(instance_root: Path, stem: str) -> Path:
    return stats_dir(instance_root) / f"s1_patch_based_scale_detection_{stem}.csv"


def global_path(instance_root: Path, stem: str) -> Path:
    return stats_dir(instance_root) / f"s1_patch_based_global_statistics_{stem}.csv"


def city_path(instance_root: Path, stem: str) -> Path:
    return stats_dir(instance_root) / f"s1_patch_based_city_statistics_{stem}.csv"


def ssl4eo_path(instance_root: Path, stem: str) -> Path:
    return stats_dir(instance_root) / f"s1_patch_based_ssl4eo_comparison_{stem}.csv"


def outlier_path(instance_root: Path, stem: str) -> Path:
    return stats_dir(instance_root) / f"s1_patch_based_outlier_diagnostics_{stem}.csv"


# ---------------------------------------------------------------------
# Row lookup helpers
# ---------------------------------------------------------------------

def key_product_band(row: Dict[str, str]) -> Tuple[str, str]:
    return str(row["product"]), str(row["band"])


def index_by_product_band(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    return {key_product_band(row): row for row in rows}


def get_row(index: Dict[Tuple[str, str], Dict[str, str]], product: str, band: str) -> Dict[str, str]:
    key = (product, band)
    if key not in index:
        fail(f"Missing row for product={product}, band={band}")
    return index[key]


# ---------------------------------------------------------------------
# Paper tables
# ---------------------------------------------------------------------

def make_main_table(
    *,
    scale_rows: List[Dict[str, str]],
    global_rows: List[Dict[str, str]],
    ssl4eo_rows: List[Dict[str, str]],
    outlier_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    scale_idx = index_by_product_band(scale_rows)
    global_idx = index_by_product_band(global_rows)
    ssl_idx = index_by_product_band(ssl4eo_rows)
    outlier_idx = index_by_product_band(outlier_rows)

    rows: List[Dict[str, object]] = []

    for product in ["SNAP-GRD", "RTC"]:
        for band in ["VV", "VH"]:
            scale = get_row(scale_idx, product, band)
            glob = get_row(global_idx, product, band)
            ssl = get_row(ssl_idx, product, band)
            out = get_row(outlier_idx, product, band)

            mean = safe_float(glob["mean"])
            median = safe_float(glob["median"])
            std = safe_float(glob["std"])
            p01 = safe_float(glob["p01"])
            p05 = safe_float(glob["p05"])
            p95 = safe_float(glob["p95"])
            p99 = safe_float(glob["p99"])

            q25 = safe_float(glob.get("p25", "0"))
            q75 = safe_float(glob.get("p75", "0"))
            iqr = q75 - q25

            ssl_mean = safe_float(ssl["ssl4eo_mean"])
            ssl_std = safe_float(ssl["ssl4eo_std"])
            mean_diff = mean - ssl_mean
            std_ratio = std / ssl_std if ssl_std != 0 else 0.0

            transformed_gt_10 = safe_float(out.get("transformed_gt_10db_percent", "0"))
            transformed_gt_0 = safe_float(out.get("transformed_gt_0db_percent", "0"))

            if str(ssl["plausibility_flag"]) == "broadly_plausible":
                interpretation = "Broadly plausible"
            elif product == "RTC":
                interpretation = "Shifted high-tail distribution; inspect before primary use"
            else:
                interpretation = "Distribution requires inspection"

            rows.append(
                {
                    "product": product,
                    "band": band,
                    "input_scale_detected": scale.get("inferred_input_scale", scale.get("inferred_scale", "")),
                    "conversion_mode": scale.get("conversion_mode", ""),
                    "scale_used": glob["scale_used"],
                    "valid_pixel_count": safe_int(glob["valid_pixel_count"]),
                    "invalid_percent": round_float(safe_float(glob["invalid_percent"]), 6),
                    "mean_db": round_float(mean, 4),
                    "median_db": round_float(median, 4),
                    "std_db": round_float(std, 4),
                    "iqr_db": round_float(iqr, 4),
                    "p01_db": round_float(p01, 4),
                    "p05_db": round_float(p05, 4),
                    "p95_db": round_float(p95, 4),
                    "p99_db": round_float(p99, 4),
                    "ssl4eo_mean_db": round_float(ssl_mean, 4),
                    "ssl4eo_std_db": round_float(ssl_std, 4),
                    "mean_difference_vs_ssl4eo_db": round_float(mean_diff, 4),
                    "std_ratio_vs_ssl4eo": round_float(std_ratio, 4),
                    "pixels_gt_0db_percent": round_float(transformed_gt_0, 4),
                    "pixels_gt_10db_percent": round_float(transformed_gt_10, 4),
                    "plausibility_flag": ssl["plausibility_flag"],
                    "paper_interpretation": interpretation,
                }
            )

    return rows


def make_snap_vs_rtc_table(main_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    idx = {
        (str(row["product"]), str(row["band"])): row
        for row in main_rows
    }

    rows: List[Dict[str, object]] = []

    for band in ["VV", "VH"]:
        snap = idx[("SNAP-GRD", band)]
        rtc = idx[("RTC", band)]

        mean_delta = safe_float(rtc["mean_db"]) - safe_float(snap["mean_db"])
        median_delta = safe_float(rtc["median_db"]) - safe_float(snap["median_db"])
        std_ratio = safe_float(rtc["std_db"]) / safe_float(snap["std_db"])
        gt10_delta = safe_float(rtc["pixels_gt_10db_percent"]) - safe_float(snap["pixels_gt_10db_percent"])

        rows.append(
            {
                "band": band,
                "snap_mean_db": snap["mean_db"],
                "rtc_mean_db": rtc["mean_db"],
                "rtc_minus_snap_mean_db": round_float(mean_delta, 4),
                "snap_median_db": snap["median_db"],
                "rtc_median_db": rtc["median_db"],
                "rtc_minus_snap_median_db": round_float(median_delta, 4),
                "snap_std_db": snap["std_db"],
                "rtc_std_db": rtc["std_db"],
                "rtc_over_snap_std_ratio": round_float(std_ratio, 4),
                "snap_pixels_gt_10db_percent": snap["pixels_gt_10db_percent"],
                "rtc_pixels_gt_10db_percent": rtc["pixels_gt_10db_percent"],
                "rtc_minus_snap_gt_10db_percent": round_float(gt10_delta, 4),
                "interpretation": (
                    "RTC is much brighter and has a much larger high-value tail than SNAP-GRD"
                    if mean_delta > 5 or gt10_delta > 5
                    else "SNAP-GRD and RTC are broadly similar"
                ),
            }
        )

    return rows


def make_city_extremes_table(
    city_rows: List[Dict[str, str]],
    *,
    top_n: int,
) -> List[Dict[str, object]]:
    out_rows: List[Dict[str, object]] = []

    for product in ["SNAP-GRD", "RTC"]:
        for band in ["VV", "VH"]:
            selected = [
                row for row in city_rows
                if row["product"] == product and row["band"] == band
            ]

            selected_sorted_high = sorted(
                selected,
                key=lambda r: safe_float(r["mean"]),
                reverse=True,
            )

            selected_sorted_low = sorted(
                selected,
                key=lambda r: safe_float(r["mean"]),
            )

            for rank, row in enumerate(selected_sorted_high[:top_n], start=1):
                out_rows.append(
                    {
                        "product": product,
                        "band": band,
                        "extreme_type": "highest_city_mean",
                        "rank": rank,
                        "city": row["city"],
                        "mean_db": round_float(safe_float(row["mean"]), 4),
                        "std_db": round_float(safe_float(row["std"]), 4),
                        "min_db": round_float(safe_float(row["min"]), 4),
                        "max_db": round_float(safe_float(row["max"]), 4),
                    }
                )

            for rank, row in enumerate(selected_sorted_low[:top_n], start=1):
                out_rows.append(
                    {
                        "product": product,
                        "band": band,
                        "extreme_type": "lowest_city_mean",
                        "rank": rank,
                        "city": row["city"],
                        "mean_db": round_float(safe_float(row["mean"]), 4),
                        "std_db": round_float(safe_float(row["std"]), 4),
                        "min_db": round_float(safe_float(row["min"]), 4),
                        "max_db": round_float(safe_float(row["max"]), 4),
                    }
                )

    return out_rows


# ---------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------

def build_latex_table(main_rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Patch-based Sentinel-1 VV/VH statistics for SNAP-GRD and RTC compared with SSL4EO-S12 reference statistics. RTC values are reported after forced $10\log_{10}(x)$ conversion.}")
    lines.append(r"\label{tab:s1_snap_rtc_ssl4eo_stats}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrrrrrrll}")
    lines.append(r"\toprule")
    lines.append(r"Product & Band & Mean & Median & Std & $p_{05}$ & $p_{95}$ & SSL4EO Mean & SSL4EO Std & $>10$ dB (\%) & Flag \\")
    lines.append(r"\midrule")

    for row in main_rows:
        lines.append(
            f"{row['product']} & "
            f"{row['band']} & "
            f"{fmt(row['mean_db'])} & "
            f"{fmt(row['median_db'])} & "
            f"{fmt(row['std_db'])} & "
            f"{fmt(row['p05_db'])} & "
            f"{fmt(row['p95_db'])} & "
            f"{fmt(row['ssl4eo_mean_db'])} & "
            f"{fmt(row['ssl4eo_std_db'])} & "
            f"{fmt(row['pixels_gt_10db_percent'])} & "
            f"{str(row['plausibility_flag']).replace('_', r'\_')} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"\noindent\textit{Note.} These are Option B patch-based statistics computed over the exact 224$\times$224 windows used by the CROMA pipeline. Because the patch stride is 112, overlapping pixels are counted more than once. The SSL4EO-S12 values are used as a broad plausibility reference rather than a strict matching target, since SSL4EO-S12 is global and multi-seasonal while this dataset is Brazil-urban/favela-focused.")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Narrative summary
# ---------------------------------------------------------------------

def build_main_conclusion(main_rows: List[Dict[str, object]], comparison_rows: List[Dict[str, object]]) -> str:
    snap_rows = [row for row in main_rows if row["product"] == "SNAP-GRD"]
    rtc_rows = [row for row in main_rows if row["product"] == "RTC"]

    snap_ok = all(row["plausibility_flag"] == "broadly_plausible" for row in snap_rows)
    rtc_ok = all(row["plausibility_flag"] == "broadly_plausible" for row in rtc_rows)

    vv_comp = next(row for row in comparison_rows if row["band"] == "VV")
    vh_comp = next(row for row in comparison_rows if row["band"] == "VH")

    if snap_ok and not rtc_ok:
        return (
            "The paper-quality patch-based statistics support SNAP-GRD as the physically more stable "
            "Sentinel-1 representation. SNAP-GRD is already in dB scale and is broadly plausible relative "
            "to SSL4EO-S12 for both VV and VH. RTC can be converted to dB using 10log10(x), but after "
            "conversion it remains strongly shifted and high-tailed: RTC is "
            f"{fmt(vv_comp['rtc_minus_snap_mean_db'])} dB brighter than SNAP-GRD for VV and "
            f"{fmt(vh_comp['rtc_minus_snap_mean_db'])} dB brighter for VH, with substantially larger "
            "standard deviation and many more pixels above 10 dB. This suggests that RTC requires "
            "diagnostic inspection before it can be treated as a physically comparable primary SAR input."
        )

    if snap_ok and rtc_ok:
        return (
            "Both SNAP-GRD and RTC appear broadly plausible relative to SSL4EO-S12 after scale handling. "
            "The two products still differ, but the differences are within a range that can be interpreted "
            "as preprocessing/domain variation rather than an obvious scale problem."
        )

    return (
        "The paper-quality statistics indicate that at least one product requires further inspection before "
        "being treated as physically comparable with SSL4EO-S12. The scale handling, outlier behaviour, and "
        "city-level extremes should be reviewed."
    )


def build_markdown_report(
    *,
    summary: Dict[str, object],
    main_rows: List[Dict[str, object]],
    comparison_rows: List[Dict[str, object]],
    city_extreme_rows: List[Dict[str, object]],
    output_paths: Dict[str, Optional[Path]],
) -> str:
    lines: List[str] = []

    lines.append("# Paper-Quality Sentinel-1 Statistics Comparison")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Created UTC: `{summary['created_utc']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Instance root: `{summary['instance_root']}`")
    lines.append(f"- Source statistics directory: `{summary['source_statistics_dir']}`")
    lines.append(f"- Output directory: `{summary['output_dir']}`")
    lines.append(f"- Patch size: `{summary['patch_size']}`")
    lines.append(f"- Stride: `{summary['stride']}`")
    lines.append(f"- Edge mode: `{summary['edge_mode']}`")
    lines.append("")
    lines.append("### Main conclusion")
    lines.append("")
    lines.append(summary["main_conclusion"])
    lines.append("")

    lines.append("## Main paper table")
    lines.append("")
    lines.append("| Product | Band | Scale used | Mean | Median | Std | p05 | p95 | SSL4EO mean | SSL4EO std | >10 dB % | Plausibility |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for row in main_rows:
        lines.append(
            f"| {row['product']} | "
            f"{row['band']} | "
            f"{row['scale_used']} | "
            f"{row['mean_db']} | "
            f"{row['median_db']} | "
            f"{row['std_db']} | "
            f"{row['p05_db']} | "
            f"{row['p95_db']} | "
            f"{row['ssl4eo_mean_db']} | "
            f"{row['ssl4eo_std_db']} | "
            f"{row['pixels_gt_10db_percent']} | "
            f"{row['plausibility_flag']} |"
        )

    lines.append("")
    lines.append("## SNAP-GRD vs RTC difference summary")
    lines.append("")
    lines.append("| Band | SNAP mean | RTC mean | RTC-SNAP mean | SNAP std | RTC std | RTC/SNAP std | SNAP >10 dB % | RTC >10 dB % | Interpretation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for row in comparison_rows:
        lines.append(
            f"| {row['band']} | "
            f"{row['snap_mean_db']} | "
            f"{row['rtc_mean_db']} | "
            f"{row['rtc_minus_snap_mean_db']} | "
            f"{row['snap_std_db']} | "
            f"{row['rtc_std_db']} | "
            f"{row['rtc_over_snap_std_ratio']} | "
            f"{row['snap_pixels_gt_10db_percent']} | "
            f"{row['rtc_pixels_gt_10db_percent']} | "
            f"{row['interpretation']} |"
        )

    lines.append("")
    lines.append("## Interpretation for paper/report")
    lines.append("")
    lines.append("The patch-based statistics show that SNAP-GRD is already in dB scale and has VV/VH means and standard deviations broadly consistent with the SSL4EO-S12 reference statistics. SNAP-GRD is brighter than SSL4EO-S12 by approximately 3 dB, which is plausible given that this dataset focuses on dense Brazilian urban/favela environments rather than a global multi-seasonal sample.")
    lines.append("")
    lines.append("RTC required forced conversion to dB using `10*log10(x)`. After this conversion, RTC became comparable in units, but its distribution remained strongly shifted toward high backscatter values and showed a much larger high-value tail than SNAP-GRD. This is visible in both the mean/std comparison and the percentage of pixels above 10 dB.")
    lines.append("")
    lines.append("Therefore, the paper-quality conclusion should not state that RTC is simply wrong. A more careful statement is that RTC is not physically comparable to SNAP-GRD without additional diagnostic inspection. The difference may be due to calibration convention, positive-valued RTC scaling, outlier influence, urban-domain effects, or remaining preprocessing artefacts.")
    lines.append("")

    lines.append("## Recommended wording")
    lines.append("")
    lines.append("> We computed patch-based Sentinel-1 VV/VH statistics over the exact 224 by 224 windows used in the CROMA experiments and compared them with SSL4EO-S12 reference statistics. SNAP-GRD was already in dB scale and showed broadly plausible statistics relative to SSL4EO-S12, with slightly brighter means consistent with the urban/favela-focused sampling. RTC was stored in a positive-valued scale and was therefore converted to dB using `10log10(x)` for comparability. After conversion, RTC remained substantially brighter and more variable than both SNAP-GRD and SSL4EO-S12, with a much larger high-value tail. This suggests that RTC requires further diagnostic inspection before being used as the primary SAR product.")
    lines.append("")

    lines.append("## City-level extremes")
    lines.append("")
    lines.append("The city-extreme table lists the cities with the highest and lowest mean values per product and band. This is intended to support the next diagnostic step, where we inspect whether RTC's high-value tail is concentrated in specific cities or spread across the dataset.")
    lines.append("")

    lines.append("## Output files")
    lines.append("")
    for key, value in output_paths.items():
        if value is not None:
            lines.append(f"- `{key}`: `{path_to_str(value)}`")

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def make_mean_std_figure(
    *,
    main_rows: List[Dict[str, object]],
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        log("WARN", "matplotlib not available; skipping mean/std figure.")
        return None

    ensure_output_can_be_written(output_path, overwrite)

    labels = []
    means = []
    stds = []
    ssl_means = []

    for row in main_rows:
        labels.append(f"{row['product']}\n{row['band']}")
        means.append(safe_float(row["mean_db"]))
        stds.append(safe_float(row["std_db"]))
        ssl_means.append(safe_float(row["ssl4eo_mean_db"]))

    x = list(range(len(labels)))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.bar(x, means, yerr=stds, capsize=4, label="Dataset mean ± std")
    ax.scatter(x, ssl_means, marker="x", s=80, label="SSL4EO-S12 mean")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Backscatter, dB")
    ax.set_title("Patch-based S1 statistics compared with SSL4EO-S12")
    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def make_outlier_figure(
    *,
    main_rows: List[Dict[str, object]],
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        log("WARN", "matplotlib not available; skipping outlier figure.")
        return None

    ensure_output_can_be_written(output_path, overwrite)

    labels = []
    values = []

    for row in main_rows:
        labels.append(f"{row['product']}\n{row['band']}")
        values.append(safe_float(row["pixels_gt_10db_percent"]))

    x = list(range(len(labels)))

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    ax.bar(x, values)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pixels > 10 dB (%)")
    ax.set_title("High-backscatter tail comparison")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def make_city_extreme_figure(
    *,
    city_extreme_rows: List[Dict[str, object]],
    band: str,
    output_path: Path,
    overwrite: bool,
) -> Optional[Path]:
    if not HAS_MATPLOTLIB:
        log("WARN", "matplotlib not available; skipping city extreme figure.")
        return None

    rows = [
        row for row in city_extreme_rows
        if row["band"] == band and row["extreme_type"] == "highest_city_mean"
    ]

    if not rows:
        return None

    ensure_output_can_be_written(output_path, overwrite)

    labels = [f"{row['product']}\n{row['city']}" for row in rows]
    values = [safe_float(row["mean_db"]) for row in rows]

    x = list(range(len(labels)))

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.bar(x, values)
    ax.axhline(SSL4EO_REFERENCE[band]["mean"], linestyle="--", linewidth=1.5, label=f"SSL4EO {band} mean")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(f"Mean {band}, dB")
    ax.set_title(f"Highest city-level mean {band} values")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-quality summary of patch-based SNAP-GRD vs RTC S1 statistics."
    )

    parser.add_argument(
        "--instance-root",
        type=Path,
        required=True,
        help="Path to instance_C_s2_nodata_repaired.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <instance-root>/metadata/croma_probing/s1_statistics_paper_quality.",
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
        "--top-n-cities",
        type=int,
        default=5,
        help="Number of city extremes to report per product/band. Default: 5.",
    )

    parser.add_argument(
        "--make-figures",
        action="store_true",
        help="Generate paper-quality figures if matplotlib is available.",
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
    stem = f"ps{args.patch_size}_st{args.stride}_{args.edge_mode}"

    source_dir = stats_dir(instance_root)

    output_dir: Path = args.output_dir or (
        instance_root
        / "metadata"
        / "croma_probing"
        / "s1_statistics_paper_quality"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    main_csv = output_dir / f"paper_s1_statistics_main_table_{stem}.csv"
    comparison_csv = output_dir / f"paper_s1_statistics_snap_vs_rtc_table_{stem}.csv"
    city_extremes_csv = output_dir / f"paper_s1_statistics_city_extremes_{stem}.csv"
    latex_path = output_dir / f"paper_s1_statistics_latex_table_{stem}.tex"
    json_path = output_dir / f"paper_s1_statistics_summary_{stem}.json"
    md_path = output_dir / f"paper_s1_statistics_summary_{stem}.md"

    fig_mean_std: Optional[Path] = None
    fig_outlier: Optional[Path] = None
    fig_city_vv: Optional[Path] = None
    fig_city_vh: Optional[Path] = None

    if args.make_figures:
        figure_dir = output_dir / "figures"
        fig_mean_std = figure_dir / f"paper_s1_mean_std_comparison_{stem}.png"
        fig_outlier = figure_dir / f"paper_s1_outlier_comparison_{stem}.png"
        fig_city_vv = figure_dir / f"paper_s1_city_mean_extremes_vv_{stem}.png"
        fig_city_vh = figure_dir / f"paper_s1_city_mean_extremes_vh_{stem}.png"

    output_paths: Dict[str, Optional[Path]] = {
        "main_table_csv": main_csv,
        "snap_vs_rtc_table_csv": comparison_csv,
        "city_extremes_csv": city_extremes_csv,
        "latex_table": latex_path,
        "json": json_path,
        "markdown": md_path,
        "figure_mean_std": fig_mean_std,
        "figure_outlier": fig_outlier,
        "figure_city_vv": fig_city_vv,
        "figure_city_vh": fig_city_vh,
    }

    log("STEP", "Building paper-quality S1 statistics summary.")
    log("INFO", f"Instance root: {path_to_str(instance_root)}")
    log("INFO", f"Source dir:     {path_to_str(source_dir)}")
    log("INFO", f"Output dir:     {path_to_str(output_dir)}")

    scale_rows = read_csv_rows(scale_path(instance_root, stem))
    global_rows = read_csv_rows(global_path(instance_root, stem))
    city_rows = read_csv_rows(city_path(instance_root, stem))
    ssl4eo_rows = read_csv_rows(ssl4eo_path(instance_root, stem))
    outlier_rows = read_csv_rows(outlier_path(instance_root, stem))

    main_rows = make_main_table(
        scale_rows=scale_rows,
        global_rows=global_rows,
        ssl4eo_rows=ssl4eo_rows,
        outlier_rows=outlier_rows,
    )

    comparison_rows = make_snap_vs_rtc_table(main_rows)

    city_extreme_rows = make_city_extremes_table(
        city_rows,
        top_n=int(args.top_n_cities),
    )

    latex_text = build_latex_table(main_rows)

    main_conclusion = build_main_conclusion(main_rows, comparison_rows)

    if args.make_figures:
        output_paths["figure_mean_std"] = make_mean_std_figure(
            main_rows=main_rows,
            output_path=fig_mean_std,
            overwrite=bool(args.overwrite),
        )

        output_paths["figure_outlier"] = make_outlier_figure(
            main_rows=main_rows,
            output_path=fig_outlier,
            overwrite=bool(args.overwrite),
        )

        output_paths["figure_city_vv"] = make_city_extreme_figure(
            city_extreme_rows=city_extreme_rows,
            band="VV",
            output_path=fig_city_vv,
            overwrite=bool(args.overwrite),
        )

        output_paths["figure_city_vh"] = make_city_extreme_figure(
            city_extreme_rows=city_extreme_rows,
            band="VH",
            output_path=fig_city_vh,
            overwrite=bool(args.overwrite),
        )

    summary_payload: Dict[str, object] = {
        "created_utc": now_utc(),
        "status": "passed",
        "instance_root": path_to_str(instance_root),
        "source_statistics_dir": path_to_str(source_dir),
        "output_dir": path_to_str(output_dir),
        "patch_size": int(args.patch_size),
        "stride": int(args.stride),
        "edge_mode": str(args.edge_mode),
        "main_conclusion": main_conclusion,
        "ssl4eo_reference": SSL4EO_REFERENCE,
        "n_main_rows": len(main_rows),
        "n_comparison_rows": len(comparison_rows),
        "n_city_extreme_rows": len(city_extreme_rows),
        "outputs": {
            key: "" if value is None else path_to_str(value)
            for key, value in output_paths.items()
        },
    }

    markdown_text = build_markdown_report(
        summary=summary_payload,
        main_rows=main_rows,
        comparison_rows=comparison_rows,
        city_extreme_rows=city_extreme_rows,
        output_paths=output_paths,
    )

    log("STEP", "Writing paper-quality outputs.")

    write_csv(
        main_csv,
        main_rows,
        overwrite=bool(args.overwrite),
    )

    write_csv(
        comparison_csv,
        comparison_rows,
        overwrite=bool(args.overwrite),
    )

    write_csv(
        city_extremes_csv,
        city_extreme_rows,
        overwrite=bool(args.overwrite),
    )

    write_text(latex_path, latex_text, overwrite=bool(args.overwrite))
    write_json(json_path, summary_payload, overwrite=bool(args.overwrite))
    write_text(md_path, markdown_text, overwrite=bool(args.overwrite))

    log("OK", f"Wrote main table CSV:      {path_to_str(main_csv)}")
    log("OK", f"Wrote SNAP-vs-RTC CSV:     {path_to_str(comparison_csv)}")
    log("OK", f"Wrote city extremes CSV:   {path_to_str(city_extremes_csv)}")
    log("OK", f"Wrote LaTeX table:         {path_to_str(latex_path)}")
    log("OK", f"Wrote JSON:                {path_to_str(json_path)}")
    log("OK", f"Wrote Markdown:            {path_to_str(md_path)}")

    for key in ["figure_mean_std", "figure_outlier", "figure_city_vv", "figure_city_vh"]:
        if output_paths.get(key) is not None:
            log("OK", f"Wrote figure {key}: {path_to_str(output_paths[key])}")

    log("STEP", "Final paper-quality S1 statistics summary.")
    log("OK", "Status: passed")
    log("OK", f"Main conclusion: {main_conclusion}")


if __name__ == "__main__":
    main()