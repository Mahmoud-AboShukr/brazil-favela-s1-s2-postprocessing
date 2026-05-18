#!/usr/bin/env python3
"""
Build selected Sentinel-1 fallback product table.

This script reads fallback candidate CSVs produced by:

    10_search_fallback_s1_products_for_missing_aoi.py

It selects one fallback Sentinel-1 GRD item per city, based on:
- overlap with missing S1 AOI
- full coverage flag
- temporal distance to ranking date
- required polarizations
- instrument mode / product type

It does not download data and does not modify rasters.

Outputs:
    qc/s1_fallback_selection/
        selected_s1_fallback_products.csv
        selected_s1_fallback_products.json
        selected_s1_fallback_products.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build selected S1 fallback product table from candidate CSVs."
    )

    parser.add_argument(
        "--instance-root",
        type=str,
        required=True,
        help=(
            "Root of repaired instance, e.g. "
            "D:/post_processing_dataset/dataset_instances/instance_C_s2_nodata_repaired"
        ),
    )

    parser.add_argument(
        "--candidate-roots",
        nargs="+",
        required=True,
        help=(
            "One or more fallback search output folders. Example: "
            "qc/s1_fallback_search qc/s1_fallback_search_expanded"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output folder. If omitted, writes to "
            "<instance-root>/qc/s1_fallback_selection"
        ),
    )

    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional city list. If omitted, cities are inferred from candidate CSVs.",
    )

    parser.add_argument(
        "--min-overlap-percent",
        type=float,
        default=95.0,
        help="Minimum overlap required for automatic selection. Default: 95.0.",
    )

    parser.add_argument(
        "--required-polarizations",
        nargs="*",
        default=["VV", "VH"],
        help="Required polarizations. Default: VV VH.",
    )

    parser.add_argument(
        "--preferred-instrument-mode",
        type=str,
        default="IW",
        help="Preferred instrument mode. Default: IW.",
    )

    parser.add_argument(
        "--preferred-product-type-contains",
        type=str,
        default="GRD",
        help="Product type should contain this string. Default: GRD.",
    )

    parser.add_argument(
        "--manual-selection-json",
        type=str,
        default=None,
        help=(
            "Optional JSON mapping city -> selected_item_id to force selections. "
            "Example: {\"campo_grande\": \"S1B_...\", \"sao_goncalo\": \"S1A_...\"}"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser.parse_args()


def resolve_path(instance_root: Path, path_text: str) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return instance_root / path


def safe_remove(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")


def read_manual_selection(path_text: str | None) -> dict[str, str]:
    if path_text is None:
        return {}

    path = Path(path_text)

    if not path.exists():
        raise FileNotFoundError(f"Manual selection JSON does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Manual selection JSON must be an object mapping city -> item_id.")

    return {str(k): str(v) for k, v in data.items()}


def discover_candidate_csvs(candidate_root: Path) -> list[Path]:
    csvs: list[Path] = []

    combined = candidate_root / "combined_s1_fallback_candidates.csv"
    if combined.exists():
        csvs.append(combined)

    per_city = candidate_root / "per_city"
    if per_city.exists():
        csvs.extend(sorted(per_city.glob("*_s1_fallback_candidates.csv")))

    # De-duplicate while preserving order.
    seen = set()
    unique = []

    for p in csvs:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def read_candidate_csv(path: Path, source_root: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if "no_candidates" in df.columns:
        return pd.DataFrame()

    required = {"city", "item_id"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()
    df["candidate_csv_path"] = str(path)
    df["candidate_search_root"] = str(source_root)

    return df


def load_all_candidates(candidate_roots: list[Path]) -> pd.DataFrame:
    frames = []

    for root in candidate_roots:
        if not root.exists():
            raise FileNotFoundError(f"Candidate root does not exist: {root}")

        csvs = discover_candidate_csvs(root)

        if not csvs:
            print(f"[WARN] No candidate CSVs found under {root}")
            continue

        for csv_path in csvs:
            df = read_candidate_csv(csv_path, source_root=root)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)

    # Remove duplicate rows for same city/item, keeping the best-looking one.
    numeric_cols = [
        "rank",
        "overlap_percent_of_missing_aoi",
        "temporal_distance_days_from_ranking_date",
    ]

    for col in numeric_cols:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    if "overlap_percent_of_missing_aoi" not in all_df.columns:
        all_df["overlap_percent_of_missing_aoi"] = 0.0

    if "temporal_distance_days_from_ranking_date" not in all_df.columns:
        all_df["temporal_distance_days_from_ranking_date"] = 999999.0

    if "rank" not in all_df.columns:
        all_df["rank"] = 999999

    all_df = all_df.sort_values(
        by=[
            "city",
            "item_id",
            "overlap_percent_of_missing_aoi",
            "temporal_distance_days_from_ranking_date",
            "rank",
        ],
        ascending=[True, True, False, True, True],
    )

    all_df = all_df.drop_duplicates(subset=["city", "item_id"], keep="first")

    return all_df


def polarizations_ok(polarizations: str, required: list[str]) -> bool:
    available = {
        p.strip().upper()
        for p in str(polarizations).replace("/", ",").replace(" ", ",").split(",")
        if p.strip()
    }

    needed = {p.upper() for p in required if p.strip()}

    return needed.issubset(available)


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def score_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "overlap_percent_of_missing_aoi",
        "temporal_distance_days_from_ranking_date",
        "rank",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "full_coverage_candidate" not in df.columns:
        df["full_coverage_candidate"] = False

    if "polarizations" not in df.columns:
        df["polarizations"] = ""

    if "instrument_mode" not in df.columns:
        df["instrument_mode"] = ""

    if "product_type" not in df.columns:
        df["product_type"] = ""

    df["polarizations_ok"] = df["polarizations"].apply(
        lambda x: polarizations_ok(str(x), args.required_polarizations)
    )

    df["instrument_mode_ok"] = (
        df["instrument_mode"].astype(str).str.upper() == args.preferred_instrument_mode.upper()
    )

    df["product_type_ok"] = df["product_type"].astype(str).str.upper().str.contains(
        args.preferred_product_type_contains.upper(),
        na=False,
    )

    df["full_coverage_bool"] = df["full_coverage_candidate"].apply(truthy)

    df["meets_min_overlap"] = df["overlap_percent_of_missing_aoi"] >= args.min_overlap_percent

    # Higher is better for positive score components.
    df["selection_score"] = (
        df["overlap_percent_of_missing_aoi"].fillna(0.0) * 1000.0
        + df["full_coverage_bool"].astype(int) * 100.0
        + df["polarizations_ok"].astype(int) * 50.0
        + df["instrument_mode_ok"].astype(int) * 25.0
        + df["product_type_ok"].astype(int) * 25.0
        - df["temporal_distance_days_from_ranking_date"].fillna(999999.0) * 0.01
    )

    return df


def select_for_city(
    city: str,
    candidates: pd.DataFrame,
    manual_selection: dict[str, str],
    args: argparse.Namespace,
) -> dict:
    city_df = candidates[candidates["city"].astype(str) == city].copy()

    if city_df.empty:
        return {
            "city": city,
            "selection_status": "no_candidates",
            "selected_item_id": "",
            "decision_reason": "No fallback candidates available for this city.",
        }

    city_df = score_candidates(city_df, args)

    if city in manual_selection:
        selected_id = manual_selection[city]
        selected_rows = city_df[city_df["item_id"].astype(str) == selected_id]

        if selected_rows.empty:
            return {
                "city": city,
                "selection_status": "manual_selection_not_found",
                "selected_item_id": selected_id,
                "decision_reason": (
                    "Manual item_id was requested but was not present in candidate tables."
                ),
            }

        row = selected_rows.iloc[0].to_dict()
        status = "selected_manual"
        reason = "Selected by manual override JSON."
    else:
        # Auto-selection: only prefer rows meeting minimum criteria.
        eligible = city_df[
            (city_df["meets_min_overlap"])
            & (city_df["polarizations_ok"])
            & (city_df["instrument_mode_ok"])
            & (city_df["product_type_ok"])
        ].copy()

        if eligible.empty:
            # Still select the top-ranked row for documentation, but flag it.
            ranked = city_df.sort_values(
                by=[
                    "overlap_percent_of_missing_aoi",
                    "selection_score",
                    "temporal_distance_days_from_ranking_date",
                ],
                ascending=[False, False, True],
            )
            row = ranked.iloc[0].to_dict()
            status = "best_available_but_below_threshold"
            reason = (
                "No candidate met all automatic thresholds. This is the best available "
                "candidate but should be reviewed before use."
            )
        else:
            ranked = eligible.sort_values(
                by=[
                    "overlap_percent_of_missing_aoi",
                    "full_coverage_bool",
                    "selection_score",
                    "temporal_distance_days_from_ranking_date",
                ],
                ascending=[False, False, False, True],
            )
            row = ranked.iloc[0].to_dict()
            status = "selected_auto"
            reason = (
                "Selected automatically: high AOI overlap, required polarizations, "
                "preferred instrument mode, and GRD product type."
            )

    selected = {
        "city": city,
        "selection_status": status,
        "decision_reason": reason,
        "selected_item_id": row.get("item_id", ""),
        "selected_datetime": row.get("datetime", ""),
        "selected_platform": row.get("platform", ""),
        "selected_constellation": row.get("constellation", ""),
        "selected_instrument_mode": row.get("instrument_mode", ""),
        "selected_product_type": row.get("product_type", ""),
        "selected_orbit_state": row.get("orbit_state", ""),
        "selected_relative_orbit": row.get("relative_orbit", ""),
        "selected_absolute_orbit": row.get("absolute_orbit", ""),
        "selected_polarizations": row.get("polarizations", ""),
        "selected_overlap_percent_of_missing_aoi": row.get("overlap_percent_of_missing_aoi", ""),
        "selected_full_coverage_candidate": row.get("full_coverage_candidate", ""),
        "selected_temporal_distance_days_from_ranking_date": row.get(
            "temporal_distance_days_from_ranking_date", ""
        ),
        "selected_ranking_date": row.get("ranking_date", ""),
        "selected_vv_asset_href": row.get("vv_asset_href", ""),
        "selected_vh_asset_href": row.get("vh_asset_href", ""),
        "selected_manifest_asset_href": row.get("manifest_asset_href", ""),
        "selected_preview_href": row.get("preview_href", ""),
        "candidate_search_root": row.get("candidate_search_root", ""),
        "candidate_csv_path": row.get("candidate_csv_path", ""),
        "candidate_note": row.get("candidate_note", ""),
        "num_candidates_for_city": int(len(city_df)),
        "num_candidates_meeting_min_overlap": int(city_df["meets_min_overlap"].sum()),
        "num_full_coverage_candidates": int(city_df["full_coverage_bool"].sum()),
        "min_overlap_percent_required": args.min_overlap_percent,
        "required_polarizations": ",".join(args.required_polarizations),
        "preferred_instrument_mode": args.preferred_instrument_mode,
        "preferred_product_type_contains": args.preferred_product_type_contains,
    }

    return selected


def write_csv(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    if not rows:
        raise ValueError("No rows to write.")

    fields = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def write_markdown(rows: list[dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {path}")

    cols = [
        "city",
        "selection_status",
        "selected_item_id",
        "selected_datetime",
        "selected_overlap_percent_of_missing_aoi",
        "selected_full_coverage_candidate",
        "selected_orbit_state",
        "selected_relative_orbit",
        "selected_polarizations",
        "selected_temporal_distance_days_from_ranking_date",
        "decision_reason",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Selected Sentinel-1 fallback products\n\n")
        f.write(
            "This table selects one fallback Sentinel-1 GRD candidate per city. "
            "These selections are intended for later download, SNAP preprocessing, "
            "alignment to the S2-filled grid, and controlled missing-pixel filling.\n\n"
        )

        f.write("## Selection counts\n\n")
        counts = {}
        for row in rows:
            status = row["selection_status"]
            counts[status] = counts.get(status, 0) + 1

        for status, count in sorted(counts.items()):
            f.write(f"- `{status}`: {count}\n")

        f.write("\n## Selected products\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")

        for row in rows:
            values = []
            for col in cols:
                value = row.get(col, "")
                if isinstance(value, float):
                    value = f"{value:.6f}"
                values.append(str(value).replace("|", "/"))
            f.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()

    instance_root = Path(args.instance_root)

    if not instance_root.exists():
        raise FileNotFoundError(f"Instance root does not exist: {instance_root}")

    candidate_roots = [resolve_path(instance_root, p) for p in args.candidate_roots]

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else instance_root / "qc" / "s1_fallback_selection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manual_selection = read_manual_selection(args.manual_selection_json)

    print(f"[INFO] Instance root: {instance_root}")
    print(f"[INFO] Candidate roots:")
    for root in candidate_roots:
        print(f"  - {root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] min_overlap_percent: {args.min_overlap_percent}")
    print(f"[INFO] required_polarizations: {args.required_polarizations}")
    print(f"[INFO] preferred_instrument_mode: {args.preferred_instrument_mode}")
    print(f"[INFO] preferred_product_type_contains: {args.preferred_product_type_contains}")

    candidates = load_all_candidates(candidate_roots)

    if candidates.empty:
        raise ValueError("No candidate rows found in the provided candidate roots.")

    if args.cities:
        cities = sorted(args.cities)
    else:
        cities = sorted(candidates["city"].astype(str).unique())

    print(f"[INFO] Loaded candidate rows: {len(candidates)}")
    print(f"[INFO] Cities to select: {cities}")

    rows = []

    for city in cities:
        selected = select_for_city(
            city=city,
            candidates=candidates,
            manual_selection=manual_selection,
            args=args,
        )
        rows.append(selected)

        print(
            "[OK] "
            f"{city}: status={selected['selection_status']} | "
            f"item={selected['selected_item_id']} | "
            f"overlap={selected['selected_overlap_percent_of_missing_aoi']}"
        )

    csv_path = output_dir / "selected_s1_fallback_products.csv"
    json_path = output_dir / "selected_s1_fallback_products.json"
    md_path = output_dir / "selected_s1_fallback_products.md"

    write_csv(rows, csv_path, overwrite=args.overwrite)
    write_json(rows, json_path, overwrite=args.overwrite)
    write_markdown(rows, md_path, overwrite=args.overwrite)

    print("\n[DONE] Wrote:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    print("\n[SUMMARY]")
    counts = {}
    for row in rows:
        status = row["selection_status"]
        counts[status] = counts.get(status, 0) + 1

    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()