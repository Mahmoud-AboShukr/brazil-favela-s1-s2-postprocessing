#!/usr/bin/env python3
"""
Build city/state/region metadata table for the Brazil favela dataset.

Purpose
-------
This script creates a clean city-level metadata table for the 26-city dataset.

It links each internal city slug to:
    - human-readable city name
    - Brazilian state name
    - state abbreviation
    - macro-region
    - dataset source group
    - availability of Instance B products:
        S2 reflectance
        SNAP S1
        final label

This table is needed before defining geographic train/validation/test splits.

Outputs
-------
CSV:
    <output_root>/metadata/city_region_table.csv

Markdown:
    <output_root>/metadata/city_region_table.md

Example
-------
    python src/favela_postprocessing/10_build_city_region_table.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml


SCRIPT_NAME = "10_build_city_region_table.py"
INSTANCE_NAME = "instance_B_standard_rs"


CITY_METADATA: Dict[str, Dict[str, str]] = {
    "belem": {
        "city_name": "Belém",
        "state": "Pará",
        "state_abbrev": "PA",
        "region": "North",
    },
    "belo_horizonte": {
        "city_name": "Belo Horizonte",
        "state": "Minas Gerais",
        "state_abbrev": "MG",
        "region": "Southeast",
    },
    "brasilia": {
        "city_name": "Brasília",
        "state": "Distrito Federal",
        "state_abbrev": "DF",
        "region": "Central-West",
    },
    "campinas": {
        "city_name": "Campinas",
        "state": "São Paulo",
        "state_abbrev": "SP",
        "region": "Southeast",
    },
    "campo_grande": {
        "city_name": "Campo Grande",
        "state": "Mato Grosso do Sul",
        "state_abbrev": "MS",
        "region": "Central-West",
    },
    "curitiba": {
        "city_name": "Curitiba",
        "state": "Paraná",
        "state_abbrev": "PR",
        "region": "South",
    },
    "duque_de_caxias": {
        "city_name": "Duque de Caxias",
        "state": "Rio de Janeiro",
        "state_abbrev": "RJ",
        "region": "Southeast",
    },
    "fortaleza": {
        "city_name": "Fortaleza",
        "state": "Ceará",
        "state_abbrev": "CE",
        "region": "Northeast",
    },
    "goiania": {
        "city_name": "Goiânia",
        "state": "Goiás",
        "state_abbrev": "GO",
        "region": "Central-West",
    },
    "guarulhos": {
        "city_name": "Guarulhos",
        "state": "São Paulo",
        "state_abbrev": "SP",
        "region": "Southeast",
    },
    "joao_pessoa": {
        "city_name": "João Pessoa",
        "state": "Paraíba",
        "state_abbrev": "PB",
        "region": "Northeast",
    },
    "maceio": {
        "city_name": "Maceió",
        "state": "Alagoas",
        "state_abbrev": "AL",
        "region": "Northeast",
    },
    "manaus": {
        "city_name": "Manaus",
        "state": "Amazonas",
        "state_abbrev": "AM",
        "region": "North",
    },
    "natal": {
        "city_name": "Natal",
        "state": "Rio Grande do Norte",
        "state_abbrev": "RN",
        "region": "Northeast",
    },
    "nova_iguacu": {
        "city_name": "Nova Iguaçu",
        "state": "Rio de Janeiro",
        "state_abbrev": "RJ",
        "region": "Southeast",
    },
    "porto_alegre": {
        "city_name": "Porto Alegre",
        "state": "Rio Grande do Sul",
        "state_abbrev": "RS",
        "region": "South",
    },
    "recife": {
        "city_name": "Recife",
        "state": "Pernambuco",
        "state_abbrev": "PE",
        "region": "Northeast",
    },
    "rio_de_janeiro": {
        "city_name": "Rio de Janeiro",
        "state": "Rio de Janeiro",
        "state_abbrev": "RJ",
        "region": "Southeast",
    },
    "salvador": {
        "city_name": "Salvador",
        "state": "Bahia",
        "state_abbrev": "BA",
        "region": "Northeast",
    },
    "santo_andre": {
        "city_name": "Santo André",
        "state": "São Paulo",
        "state_abbrev": "SP",
        "region": "Southeast",
    },
    "sao_bernardo_do_campo": {
        "city_name": "São Bernardo do Campo",
        "state": "São Paulo",
        "state_abbrev": "SP",
        "region": "Southeast",
    },
    "sao_goncalo": {
        "city_name": "São Gonçalo",
        "state": "Rio de Janeiro",
        "state_abbrev": "RJ",
        "region": "Southeast",
    },
    "sao_luis": {
        "city_name": "São Luís",
        "state": "Maranhão",
        "state_abbrev": "MA",
        "region": "Northeast",
    },
    "sao_paulo": {
        "city_name": "São Paulo",
        "state": "São Paulo",
        "state_abbrev": "SP",
        "region": "Southeast",
    },
    "sorocaba": {
        "city_name": "Sorocaba",
        "state": "São Paulo",
        "state_abbrev": "SP",
        "region": "Southeast",
    },
    "teresina": {
        "city_name": "Teresina",
        "state": "Piauí",
        "state_abbrev": "PI",
        "region": "Northeast",
    },
}


REGION_ORDER = {
    "North": 1,
    "Northeast": 2,
    "Central-West": 3,
    "Southeast": 4,
    "South": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build city-region metadata table for the Brazil favela dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--city",
        action="append",
        default=None,
        help="Include only one city. Can be repeated.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "output_root" not in cfg:
        raise KeyError("Missing required key in config: output_root")

    return cfg


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_city_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def discover_official_cities(
    output_root: Path,
    selected_cities: Optional[Sequence[str]],
) -> List[str]:
    if selected_cities:
        return sorted(set(normalize_city_name(city) for city in selected_cities))

    s2_final_root = output_root / "s2_final"

    if not s2_final_root.exists():
        raise FileNotFoundError(
            f"Cannot discover official city list because this folder does not exist: {s2_final_root}"
        )

    cities = sorted(
        path.name
        for path in s2_final_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )

    if not cities:
        raise RuntimeError(f"No city folders found under: {s2_final_root}")

    return cities


def s2_reflectance_path(output_root: Path, city: str) -> Path:
    return (
        output_root
        / "dataset_instances"
        / INSTANCE_NAME
        / "s2"
        / city
        / f"{city}_s2_12bands_reflectance_10m.tif"
    )


def s1_snap_path(output_root: Path, city: str) -> Path:
    return (
        output_root
        / "dataset_instances"
        / INSTANCE_NAME
        / "s1_snap"
        / city
        / f"{city}_s1_snap_vv_vh_vvdiff_10m_aligned.tif"
    )


def label_path(output_root: Path, city: str) -> Path:
    return output_root / "labels_final" / city / f"{city}_label_final.tif"


def original_polygon_mask_path(output_root: Path, city: str) -> Path:
    return (
        output_root
        / "auxiliary"
        / "original_polygon_masks"
        / city
        / f"{city}_original_polygon_mask.tif"
    )


def instance_b_summary_path(output_root: Path) -> Path:
    return output_root / "qc" / "instance_B_standard_rs_summary.csv"


def read_instance_b_status(output_root: Path) -> Dict[str, bool]:
    """
    Read the Instance B summary if available.

    If unavailable, the caller will still compute availability from file existence.
    """
    path = instance_b_summary_path(output_root)

    if not path.exists():
        return {}

    df = pd.read_csv(path)

    if "city" not in df.columns or "complete_instance_B_standard_rs" not in df.columns:
        return {}

    status = {}

    for _, row in df.iterrows():
        city = normalize_city_name(row["city"])
        value = row["complete_instance_B_standard_rs"]

        if isinstance(value, str):
            status[city] = value.strip().lower() in {"true", "1", "yes"}
        else:
            status[city] = bool(value)

    return status


def build_row(output_root: Path, city: str, instance_b_status: Dict[str, bool]) -> Dict[str, Any]:
    city = normalize_city_name(city)

    metadata = CITY_METADATA.get(city)

    if metadata is None:
        metadata = {
            "city_name": city,
            "state": "UNKNOWN",
            "state_abbrev": "UNKNOWN",
            "region": "UNKNOWN",
        }

    s2_path = s2_reflectance_path(output_root, city)
    s1_path = s1_snap_path(output_root, city)
    lab_path = label_path(output_root, city)
    orig_path = original_polygon_mask_path(output_root, city)

    has_s2_reflectance = s2_path.exists()
    has_s1_snap = s1_path.exists()
    has_label = lab_path.exists()
    has_original_polygon_mask = orig_path.exists()

    complete_instance_b = instance_b_status.get(
        city,
        bool(has_s2_reflectance and has_s1_snap and has_label),
    )

    region = metadata["region"]

    return {
        "city": city,
        "city_name": metadata["city_name"],
        "state": metadata["state"],
        "state_abbrev": metadata["state_abbrev"],
        "region": region,
        "region_order": REGION_ORDER.get(region, 999),
        "source_group": "brazil_26_city_favela_dataset",
        "dataset_instance": INSTANCE_NAME,
        "has_s2_reflectance": has_s2_reflectance,
        "has_s1_snap": has_s1_snap,
        "has_label": has_label,
        "has_original_polygon_mask": has_original_polygon_mask,
        "complete_instance_B_standard_rs": complete_instance_b,
        "s2_reflectance_path": str(s2_path),
        "s1_snap_path": str(s1_path),
        "label_path": str(lab_path),
        "original_polygon_mask_path": str(orig_path),
    }


def validate_rows(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []

    expected_city_count = 26

    if len(df) != expected_city_count:
        errors.append(f"Expected {expected_city_count} cities, found {len(df)}.")

    unknown = df[df["region"] == "UNKNOWN"]

    if not unknown.empty:
        errors.append(
            "Some cities have UNKNOWN metadata: "
            + ", ".join(unknown["city"].astype(str).tolist())
        )

    required_regions = {"North", "Northeast", "Central-West", "Southeast", "South"}
    found_regions = set(df["region"].dropna().astype(str).unique())

    missing_regions = sorted(required_regions - found_regions)

    if missing_regions:
        errors.append(f"Missing Brazilian macro-regions: {missing_regions}")

    if not df["complete_instance_B_standard_rs"].all():
        incomplete = df.loc[
            ~df["complete_instance_B_standard_rs"],
            "city",
        ].astype(str).tolist()

        errors.append(
            "Some cities are not complete for Instance B: "
            + ", ".join(incomplete)
        )

    duplicate_cities = df[df["city"].duplicated()]["city"].tolist()

    if duplicate_cities:
        errors.append(
            "Duplicate city rows found: "
            + ", ".join(duplicate_cities)
        )

    return errors


def markdown_escape(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return "nan"
    except Exception:
        pass

    text = str(value)
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


def df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    columns = list(df.columns)

    header = "| " + " | ".join(markdown_escape(col) for col in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    rows = []

    for _, row in df.iterrows():
        rows.append(
            "| "
            + " | ".join(markdown_escape(row[col]) for col in columns)
            + " |"
        )

    return "\n".join([header, separator] + rows)


def write_markdown_summary(df: pd.DataFrame, md_path: Path, errors: List[str]) -> None:
    ensure_dir(md_path.parent)

    lines: List[str] = []

    lines.append("# City Region Table")
    lines.append("")
    lines.append(f"Generated by `{SCRIPT_NAME}`.")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "This table provides the city-level geographic metadata required for "
        "dataset splitting, patch indexing, regional evaluation, and later ML-ready exports."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    summary = pd.DataFrame(
        [
            {"metric": "Number of cities", "value": len(df)},
            {"metric": "Regions represented", "value": df["region"].nunique()},
            {"metric": "States represented", "value": df["state_abbrev"].nunique()},
            {"metric": "Cities complete for Instance B", "value": int(df["complete_instance_B_standard_rs"].sum())},
            {"metric": "Cities with S2 reflectance", "value": int(df["has_s2_reflectance"].sum())},
            {"metric": "Cities with SNAP S1", "value": int(df["has_s1_snap"].sum())},
            {"metric": "Cities with labels", "value": int(df["has_label"].sum())},
        ]
    )

    lines.append(df_to_markdown_table(summary))
    lines.append("")

    lines.append("## Cities per region")
    lines.append("")

    region_counts = (
        df.groupby(["region_order", "region"])
        .size()
        .reset_index(name="city_count")
        .sort_values(["region_order", "region"])
        .drop(columns=["region_order"])
    )

    lines.append(df_to_markdown_table(region_counts))
    lines.append("")

    lines.append("## Cities per state")
    lines.append("")

    state_counts = (
        df.groupby(["state_abbrev", "state"])
        .size()
        .reset_index(name="city_count")
        .sort_values(["state_abbrev"])
    )

    lines.append(df_to_markdown_table(state_counts))
    lines.append("")

    lines.append("## City table")
    lines.append("")

    compact_cols = [
        "city",
        "city_name",
        "state_abbrev",
        "state",
        "region",
        "complete_instance_B_standard_rs",
    ]

    lines.append(df_to_markdown_table(df[compact_cols]))
    lines.append("")

    lines.append("## Validation")
    lines.append("")

    if errors:
        lines.append("Validation errors/warnings:")
        lines.append("")
        for error in errors:
            lines.append(f"- {error}")
    else:
        lines.append("No validation errors found.")

    lines.append("")
    lines.append("## Recommended next step")
    lines.append("")
    lines.append(
        "Use this table to generate geographic split definitions, especially "
        "one-city-per-region validation/test splits and leave-one-region-out folds."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))
    cities = discover_official_cities(output_root, args.city)

    metadata_root = output_root / "metadata"
    ensure_dir(metadata_root)

    csv_path = metadata_root / "city_region_table.csv"
    md_path = metadata_root / "city_region_table.md"

    print("[INFO] Build city-region table")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Cities selected: {len(cities)}")
    print(f"[INFO] CSV output: {csv_path}")
    print(f"[INFO] Markdown output: {md_path}")

    instance_b_status = read_instance_b_status(output_root)

    rows = [
        build_row(
            output_root=output_root,
            city=city,
            instance_b_status=instance_b_status,
        )
        for city in cities
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values(["region_order", "state_abbrev", "city"]).reset_index(drop=True)

    errors = validate_rows(df)

    df.to_csv(csv_path, index=False)
    write_markdown_summary(df, md_path, errors)

    print(f"[INFO] Wrote CSV: {csv_path}")
    print(f"[INFO] Wrote Markdown summary: {md_path}")

    print("[INFO] Cities per region:")
    print(
        df.groupby("region")
        .size()
        .sort_index()
        .rename("city_count")
        .to_string()
    )

    print("[INFO] Complete Instance B cities:")
    print(f"{int(df['complete_instance_B_standard_rs'].sum())} / {len(df)}")

    if errors:
        print("[WARN] Validation messages:")
        for error in errors:
            print(f"       - {error}")
        return 1

    print("[INFO] City-region table is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())