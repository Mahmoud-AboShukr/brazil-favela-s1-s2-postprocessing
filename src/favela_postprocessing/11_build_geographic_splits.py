#!/usr/bin/env python3
"""
Build geographic split definitions for the Brazil favela dataset.

Purpose
-------
This script creates deterministic city-level geographic split protocols using:

    <output_root>/metadata/city_region_table.csv

It produces split files that will later be joined to patch metadata.

Why this matters
----------------
Random patch splits can leak spatial information because nearby patches from the same
city share acquisition conditions, urban morphology, road patterns, texture, and label
structure. City-level and region-level splits are more realistic tests of geographic
generalization.

Split strategies
----------------

1. train_covered_region_test
   Recommended main split.

   - Test set has one city from each Brazilian macro-region.
   - Training set keeps at least one city from every macro-region.
   - Validation uses one city from regions with enough cities.
   - This avoids the problem where North/South disappear from training.

2. balanced_5val_5test
   Diagnostic split.

   - Validation has one city per region.
   - Test has one city per region.
   - Train has 16 cities.
   - Because North and South only have 2 cities each, this split leaves no North/South
     city in training. Therefore it is useful as a diagnostic, but not recommended as
     the main training protocol.

3. leave_one_region_out
   Strongest geographic generalization protocol.

   - For each fold, one full macro-region is held out for testing.
   - Validation cities are selected from the remaining regions.
   - Remaining cities are used for training.

4. all_cities_train
   Utility split.

   - All complete cities are assigned to train.
   - Useful for final model fitting after experimental choices are fixed.

Outputs
-------
Metadata files:
    <output_root>/metadata/split_train_covered_region_test.csv
    <output_root>/metadata/split_balanced_5val_5test.csv
    <output_root>/metadata/split_leave_one_region_out.csv
    <output_root>/metadata/split_all_cities_train.csv
    <output_root>/metadata/geographic_splits_summary.md

QC file:
    <output_root>/qc/geographic_splits_summary.csv

Example
-------
    python src/favela_postprocessing/11_build_geographic_splits.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml


SCRIPT_NAME = "11_build_geographic_splits.py"


REGION_ORDER = {
    "North": 1,
    "Northeast": 2,
    "Central-West": 3,
    "Southeast": 4,
    "South": 5,
}


# Main held-out test city per region.
# These choices intentionally include geographically important / representative cases.
REGION_TEST_CITY = {
    "North": "manaus",
    "Northeast": "recife",
    "Central-West": "brasilia",
    "Southeast": "rio_de_janeiro",
    "South": "porto_alegre",
}


# Validation city per region for the diagnostic 5-val/5-test split.
REGION_VAL_CITY_BALANCED = {
    "North": "belem",
    "Northeast": "salvador",
    "Central-West": "campo_grande",
    "Southeast": "sao_paulo",
    "South": "curitiba",
}


# Validation city per region for the recommended train-covered split.
# North and South are intentionally omitted because they only have two cities each:
# one is used for test, one must remain in train.
REGION_VAL_CITY_TRAIN_COVERED = {
    "Northeast": "salvador",
    "Central-West": "campo_grande",
    "Southeast": "sao_paulo",
}


# Priority order for selecting validation cities in leave-one-region-out folds.
# The function falls back to sorted city order if a preferred city is unavailable.
REGION_VAL_PRIORITY = {
    "North": ["belem", "manaus"],
    "Northeast": ["salvador", "fortaleza", "recife", "maceio", "natal", "joao_pessoa", "sao_luis", "teresina"],
    "Central-West": ["campo_grande", "goiania", "brasilia"],
    "Southeast": ["sao_paulo", "rio_de_janeiro", "belo_horizonte", "campinas", "guarulhos", "santo_andre", "sao_bernardo_do_campo", "sorocaba", "duque_de_caxias", "nova_iguacu", "sao_goncalo"],
    "South": ["curitiba", "porto_alegre"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build geographic city-level split definitions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow cities that are not complete for Instance B. Default: false.",
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


def bool_from_any(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def load_city_region_table(output_root: Path, allow_incomplete: bool) -> pd.DataFrame:
    path = output_root / "metadata" / "city_region_table.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"City-region table not found: {path}\n"
            f"Run 10_build_city_region_table.py first."
        )

    df = pd.read_csv(path)

    required = [
        "city",
        "city_name",
        "state",
        "state_abbrev",
        "region",
        "region_order",
        "complete_instance_B_standard_rs",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"City-region table is missing required columns: {missing}")

    df = df.copy()
    df["city"] = df["city"].map(normalize_city_name)
    df["complete_instance_B_standard_rs"] = df["complete_instance_B_standard_rs"].map(bool_from_any)

    if not allow_incomplete:
        incomplete = df.loc[~df["complete_instance_B_standard_rs"], "city"].tolist()

        if incomplete:
            raise RuntimeError(
                "Some cities are not complete for Instance B. "
                "Use --allow-incomplete only if this is intentional. "
                f"Incomplete cities: {incomplete}"
            )

        df = df[df["complete_instance_B_standard_rs"]].copy()

    df = df.sort_values(["region_order", "state_abbrev", "city"]).reset_index(drop=True)

    return df


def add_common_split_columns(
    split_df: pd.DataFrame,
    strategy: str,
    strategy_description: str,
    fold_id: str = "main",
    heldout_region: str = "",
    recommended_for_main_benchmark: bool = False,
    warning: str = "",
) -> pd.DataFrame:
    out = split_df.copy()

    out.insert(0, "split_strategy", strategy)
    out.insert(1, "fold_id", fold_id)
    out.insert(2, "heldout_region", heldout_region)

    out["strategy_description"] = strategy_description
    out["recommended_for_main_benchmark"] = recommended_for_main_benchmark
    out["warning"] = warning

    role_order = {
        "train": 1,
        "val": 2,
        "test": 3,
    }

    out["split_role_order"] = out["split"].map(role_order).fillna(999).astype(int)
    out = out.sort_values(["split_role_order", "region_order", "state_abbrev", "city"]).reset_index(drop=True)

    return out


def require_city_exists(df: pd.DataFrame, city: str, context: str) -> None:
    if city not in set(df["city"].tolist()):
        raise RuntimeError(f"City '{city}' required by {context} is not in city_region_table.csv")


def build_train_covered_region_test_split(city_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recommended main split.

    It keeps training coverage for all five macro-regions while still testing on
    one city per region.
    """
    for region, city in REGION_TEST_CITY.items():
        require_city_exists(city_df, city, f"REGION_TEST_CITY[{region}]")

    for region, city in REGION_VAL_CITY_TRAIN_COVERED.items():
        require_city_exists(city_df, city, f"REGION_VAL_CITY_TRAIN_COVERED[{region}]")

    test_cities = set(REGION_TEST_CITY.values())
    val_cities = set(REGION_VAL_CITY_TRAIN_COVERED.values())

    rows: List[Dict[str, Any]] = []

    for _, row in city_df.iterrows():
        city = row["city"]

        if city in test_cities:
            split = "test"
        elif city in val_cities:
            split = "val"
        else:
            split = "train"

        item = row.to_dict()
        item["split"] = split
        rows.append(item)

    out = pd.DataFrame(rows)

    return add_common_split_columns(
        out,
        strategy="train_covered_region_test",
        strategy_description=(
            "Recommended city-level split: test has one city per region, "
            "while training retains at least one city from every region."
        ),
        fold_id="main",
        heldout_region="",
        recommended_for_main_benchmark=True,
        warning=(
            "Validation does not include North/South because those regions only have two cities; "
            "one is used for test and one is kept in train."
        ),
    )


def build_balanced_5val_5test_split(city_df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnostic split with one validation and one test city per region.

    This produces 16 train, 5 val, 5 test, but train has no North/South cities.
    """
    for region, city in REGION_TEST_CITY.items():
        require_city_exists(city_df, city, f"REGION_TEST_CITY[{region}]")

    for region, city in REGION_VAL_CITY_BALANCED.items():
        require_city_exists(city_df, city, f"REGION_VAL_CITY_BALANCED[{region}]")

    test_cities = set(REGION_TEST_CITY.values())
    val_cities = set(REGION_VAL_CITY_BALANCED.values())

    overlap = sorted(test_cities & val_cities)

    if overlap:
        raise RuntimeError(f"Balanced val/test city definitions overlap: {overlap}")

    rows: List[Dict[str, Any]] = []

    for _, row in city_df.iterrows():
        city = row["city"]

        if city in test_cities:
            split = "test"
        elif city in val_cities:
            split = "val"
        else:
            split = "train"

        item = row.to_dict()
        item["split"] = split
        rows.append(item)

    out = pd.DataFrame(rows)

    return add_common_split_columns(
        out,
        strategy="balanced_5val_5test",
        strategy_description=(
            "Diagnostic split: validation and test each contain one city per region."
        ),
        fold_id="main",
        heldout_region="",
        recommended_for_main_benchmark=False,
        warning=(
            "Diagnostic only: because North and South have only two cities each, "
            "this split leaves no North/South city in training."
        ),
    )


def choose_validation_city_for_region(region_df: pd.DataFrame, region: str) -> Optional[str]:
    """
    Choose one validation city for a given region from available training-region cities.
    """
    available = set(region_df["city"].tolist())

    for candidate in REGION_VAL_PRIORITY.get(region, []):
        if candidate in available:
            return candidate

    if not region_df.empty:
        return str(region_df.sort_values("city").iloc[0]["city"])

    return None


def build_leave_one_region_out_splits(city_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    regions = (
        city_df[["region_order", "region"]]
        .drop_duplicates()
        .sort_values(["region_order", "region"])
        ["region"]
        .tolist()
    )

    for region in regions:
        fold_id = f"leave_out_{region.lower().replace('-', '_').replace(' ', '_')}"

        fold_df = city_df.copy()
        fold_df["split"] = "train"

        fold_df.loc[fold_df["region"] == region, "split"] = "test"

        remaining_regions = [
            r for r in regions
            if r != region
        ]

        for val_region in remaining_regions:
            candidate_pool = fold_df[
                (fold_df["region"] == val_region)
                & (fold_df["split"] == "train")
            ].copy()

            val_city = choose_validation_city_for_region(candidate_pool, val_region)

            if val_city is not None:
                fold_df.loc[fold_df["city"] == val_city, "split"] = "val"

        fold_df = add_common_split_columns(
            fold_df,
            strategy="leave_one_region_out",
            strategy_description=(
                "Leave-one-region-out fold: all cities from one macro-region are used as test; "
                "validation cities are selected from the remaining regions."
            ),
            fold_id=fold_id,
            heldout_region=region,
            recommended_for_main_benchmark=True,
            warning="",
        )

        rows.extend(fold_df.to_dict(orient="records"))

    return pd.DataFrame(rows)


def build_all_cities_train_split(city_df: pd.DataFrame) -> pd.DataFrame:
    out = city_df.copy()
    out["split"] = "train"

    return add_common_split_columns(
        out,
        strategy="all_cities_train",
        strategy_description=(
            "Utility split: all complete cities are assigned to train. "
            "Useful for final model fitting after benchmark decisions are fixed."
        ),
        fold_id="main",
        heldout_region="",
        recommended_for_main_benchmark=False,
        warning="No validation/test set. Do not use for benchmark reporting.",
    )


def summarize_split(split_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["split_strategy", "fold_id", "heldout_region", "split"]

    summary = (
        split_df.groupby(group_cols)
        .agg(
            city_count=("city", "count"),
            regions_present=("region", lambda x: ",".join(sorted(set(x.astype(str))))),
            states_present=("state_abbrev", lambda x: ",".join(sorted(set(x.astype(str))))),
            cities=("city", lambda x: ",".join(sorted(x.astype(str)))),
        )
        .reset_index()
        .sort_values(["split_strategy", "fold_id", "split"])
    )

    return summary


def validate_single_strategy_split(
    split_df: pd.DataFrame,
    strategy_name: str,
    expected_city_count: int,
) -> List[str]:
    errors: List[str] = []

    if len(split_df) != expected_city_count:
        errors.append(
            f"{strategy_name}: expected {expected_city_count} rows, found {len(split_df)}."
        )

    duplicated = split_df[split_df["city"].duplicated()]["city"].tolist()

    if duplicated:
        errors.append(f"{strategy_name}: duplicated city assignments: {duplicated}")

    missing_roles = sorted(set(["train", "val", "test"]) - set(split_df["split"].unique()))

    if missing_roles and strategy_name != "all_cities_train":
        errors.append(f"{strategy_name}: missing split roles: {missing_roles}")

    return errors


def validate_leave_one_region_out(split_df: pd.DataFrame, city_df: pd.DataFrame) -> List[str]:
    errors: List[str] = []

    expected_city_count = len(city_df)
    expected_regions = set(city_df["region"].unique())

    for fold_id, fold_df in split_df.groupby("fold_id"):
        if len(fold_df) != expected_city_count:
            errors.append(f"{fold_id}: expected {expected_city_count} rows, found {len(fold_df)}.")

        duplicated = fold_df[fold_df["city"].duplicated()]["city"].tolist()

        if duplicated:
            errors.append(f"{fold_id}: duplicated city assignments: {duplicated}")

        heldout_regions = set(fold_df["heldout_region"].dropna().astype(str).unique())

        if len(heldout_regions) != 1:
            errors.append(f"{fold_id}: expected exactly one heldout_region, found {heldout_regions}")
            continue

        heldout_region = list(heldout_regions)[0]

        test_regions = set(fold_df.loc[fold_df["split"] == "test", "region"].unique())

        if test_regions != {heldout_region}:
            errors.append(
                f"{fold_id}: test regions {test_regions} do not match heldout region {heldout_region}."
            )

        non_test_regions = set(fold_df.loc[fold_df["split"] != "test", "region"].unique())

        if heldout_region in non_test_regions:
            errors.append(f"{fold_id}: heldout region appears outside test split.")

        train_regions = set(fold_df.loc[fold_df["split"] == "train", "region"].unique())
        val_regions = set(fold_df.loc[fold_df["split"] == "val", "region"].unique())

        if not train_regions:
            errors.append(f"{fold_id}: train split has no regions.")

        if not val_regions:
            errors.append(f"{fold_id}: val split has no regions.")

        if heldout_region not in expected_regions:
            errors.append(f"{fold_id}: unknown heldout region {heldout_region}.")

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


def write_markdown_summary(
    md_path: Path,
    city_df: pd.DataFrame,
    train_covered: pd.DataFrame,
    balanced: pd.DataFrame,
    loro: pd.DataFrame,
    all_train: pd.DataFrame,
    summary_df: pd.DataFrame,
    validation_messages: List[str],
) -> None:
    ensure_dir(md_path.parent)

    lines: List[str] = []

    lines.append("# Geographic Split Definitions")
    lines.append("")
    lines.append(f"Generated by `{SCRIPT_NAME}`.")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append(
        "This document defines deterministic city-level geographic splits for "
        "the Brazil favela segmentation dataset. These splits are intended to "
        "support fair evaluation of geographic generalization."
    )
    lines.append("")
    lines.append("## Why geographic splits are necessary")
    lines.append("")
    lines.append(
        "Random patch splits can leak spatial information because neighboring "
        "patches from the same city often share the same acquisition conditions, "
        "urban morphology, road networks, textures, and label context. "
        "City-level and region-level splits provide a more realistic test of "
        "model transfer to unseen cities and unseen regions."
    )
    lines.append("")
    lines.append("## Split strategies")
    lines.append("")
    lines.append("### 1. train_covered_region_test")
    lines.append("")
    lines.append(
        "Recommended main split. The test split has one city from every macro-region, "
        "while the training split retains at least one city from every macro-region. "
        "Validation is selected only from regions with enough cities."
    )
    lines.append("")
    lines.append("### 2. balanced_5val_5test")
    lines.append("")
    lines.append(
        "Diagnostic split. Validation and test each contain one city per region. "
        "However, because North and South have only two cities each, this split "
        "leaves no North/South city in training. It should not be the default main split."
    )
    lines.append("")
    lines.append("### 3. leave_one_region_out")
    lines.append("")
    lines.append(
        "Strongest geographic generalization protocol. Each fold holds out one full "
        "macro-region as the test region."
    )
    lines.append("")
    lines.append("### 4. all_cities_train")
    lines.append("")
    lines.append(
        "Utility split for final model fitting after experimental decisions are fixed. "
        "It has no validation/test set and should not be used for benchmark reporting."
    )
    lines.append("")

    lines.append("## City counts by region")
    lines.append("")

    region_counts = (
        city_df.groupby(["region_order", "region"])
        .size()
        .reset_index(name="city_count")
        .sort_values(["region_order", "region"])
        .drop(columns=["region_order"])
    )

    lines.append(df_to_markdown_table(region_counts))
    lines.append("")

    lines.append("## Split summary")
    lines.append("")

    display_summary = summary_df[
        [
            "split_strategy",
            "fold_id",
            "heldout_region",
            "split",
            "city_count",
            "regions_present",
            "cities",
        ]
    ].copy()

    lines.append(df_to_markdown_table(display_summary))
    lines.append("")

    lines.append("## Recommended main split city assignments")
    lines.append("")

    main_cols = [
    "split",
    "city",
    "city_name",
    "state_abbrev",
    "region",
    ]

    main_split = (
        train_covered
        .sort_values(["split_role_order", "region_order", "city"])
        [main_cols]
    )

    lines.append(df_to_markdown_table(main_split))
    lines.append("")

    lines.append("## Validation")
    lines.append("")

    if validation_messages:
        lines.append("Validation messages:")
        lines.append("")
        for message in validation_messages:
            lines.append(f"- {message}")
    else:
        lines.append("No validation errors found.")

    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append(
        "Use these split files when building the tiling index and patch metadata. "
        "Each patch should inherit its city-level split assignment from one of these protocols."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    output_root = Path(str(cfg["output_root"]))
    metadata_root = output_root / "metadata"
    qc_root = output_root / "qc"

    ensure_dir(metadata_root)
    ensure_dir(qc_root)

    print("[INFO] Build geographic split definitions")
    print(f"[INFO] Script: {SCRIPT_NAME}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Output root: {output_root}")

    city_df = load_city_region_table(
        output_root=output_root,
        allow_incomplete=args.allow_incomplete,
    )

    print(f"[INFO] Complete cities available for splitting: {len(city_df)}")

    train_covered = build_train_covered_region_test_split(city_df)
    balanced = build_balanced_5val_5test_split(city_df)
    loro = build_leave_one_region_out_splits(city_df)
    all_train = build_all_cities_train_split(city_df)

    train_covered_path = metadata_root / "split_train_covered_region_test.csv"
    balanced_path = metadata_root / "split_balanced_5val_5test.csv"
    loro_path = metadata_root / "split_leave_one_region_out.csv"
    all_train_path = metadata_root / "split_all_cities_train.csv"
    summary_path = qc_root / "geographic_splits_summary.csv"
    md_path = metadata_root / "geographic_splits_summary.md"

    train_covered.to_csv(train_covered_path, index=False)
    balanced.to_csv(balanced_path, index=False)
    loro.to_csv(loro_path, index=False)
    all_train.to_csv(all_train_path, index=False)

    all_splits = pd.concat(
        [
            train_covered,
            balanced,
            loro,
            all_train,
        ],
        ignore_index=True,
    )

    summary_df = summarize_split(all_splits)
    summary_df.to_csv(summary_path, index=False)

    validation_messages: List[str] = []

    validation_messages.extend(
        validate_single_strategy_split(
            train_covered,
            "train_covered_region_test",
            expected_city_count=len(city_df),
        )
    )

    validation_messages.extend(
        validate_single_strategy_split(
            balanced,
            "balanced_5val_5test",
            expected_city_count=len(city_df),
        )
    )

    validation_messages.extend(
        validate_leave_one_region_out(
            loro,
            city_df,
        )
    )

    validation_messages.extend(
        validate_single_strategy_split(
            all_train,
            "all_cities_train",
            expected_city_count=len(city_df),
        )
    )

    write_markdown_summary(
        md_path=md_path,
        city_df=city_df,
        train_covered=train_covered,
        balanced=balanced,
        loro=loro,
        all_train=all_train,
        summary_df=summary_df,
        validation_messages=validation_messages,
    )

    print(f"[INFO] Wrote: {train_covered_path}")
    print(f"[INFO] Wrote: {balanced_path}")
    print(f"[INFO] Wrote: {loro_path}")
    print(f"[INFO] Wrote: {all_train_path}")
    print(f"[INFO] Wrote: {summary_path}")
    print(f"[INFO] Wrote: {md_path}")

    print("[INFO] Split counts:")
    print(
        summary_df[
            [
                "split_strategy",
                "fold_id",
                "heldout_region",
                "split",
                "city_count",
            ]
        ].to_string(index=False)
    )

    if validation_messages:
        print("[ERROR] Validation messages:")
        for message in validation_messages:
            print(f"        - {message}")
        return 1

    print("[INFO] Geographic split definitions built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())