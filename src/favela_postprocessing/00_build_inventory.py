from pathlib import Path
import argparse
import yaml
import pandas as pd


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize_city_name(name: str) -> str:
    """
    Normalize city folder names such as:
    brasilia_V3 -> brasilia
    rio_de_janeiro_v3 -> rio_de_janeiro
    belem_ocm_v1 -> belem
    """
    name = name.lower().strip()

    suffixes = [
        "_ocm_v1",
        "_v1", "_v2", "_v3", "_v4", "_v5", "_v6", "_v7",
    ]

    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True

    return name.strip("_")


def find_s2_composite(city_dir: Path):
    """
    Find the preferred Sentinel-2 composite inside a city folder.
    Prefer small-holes-filled composites when available.
    """
    preferred = [
        "city_composite_2022_smallholes_filled.tif",
        "city_composite_2022.tif",
    ]

    for filename in preferred:
        candidate = city_dir / filename
        if candidate.exists():
            return candidate

    matches = sorted(city_dir.rglob("*composite*.tif"))
    return matches[0] if matches else None


def find_s2_cloudmask(city_dir: Path):
    matches = sorted(city_dir.rglob("*cloudmask*.tif"))
    return matches[0] if matches else None


def find_s2_diagnostics(city_dir: Path):
    matches = sorted(city_dir.rglob("*diagnostics*.json"))
    return matches[0] if matches else None


def find_s1_product(s1_city_dir: Path):
    """
    Find a Sentinel-1 product for a city.

    We search recursively because the exact folder structure may differ
    between cities.
    """
    if not s1_city_dir.exists():
        return None

    matches = sorted(s1_city_dir.rglob("*.tif"))
    if not matches:
        return None

    priority_words = ["aligned", "rtc", "vv", "vh"]

    scored = []
    for path in matches:
        score = sum(word in path.name.lower() for word in priority_words)
        scored.append((score, str(path), path))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def main():
    parser = argparse.ArgumentParser(
        description="Build city-level input inventory for S1/S2/favela post-processing."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    s2_root = Path(cfg["s2_root"])
    s1_root = Path(cfg["s1_root"])
    output_root = Path(cfg["output_root"])

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    if not s2_root.exists():
        raise FileNotFoundError(f"S2 root does not exist: {s2_root}")

    for city_dir in sorted(s2_root.iterdir()):
        if not city_dir.is_dir():
            continue

        s2_path = find_s2_composite(city_dir)
        if s2_path is None:
            continue

        city = normalize_city_name(city_dir.name)

        s2_cloudmask_path = find_s2_cloudmask(city_dir)
        s2_diagnostics_path = find_s2_diagnostics(city_dir)

        s1_city_dir = s1_root / city
        s1_path = find_s1_product(s1_city_dir)

        rows.append(
            {
                "city": city,
                "s2_folder_name": city_dir.name,
                "s2_city_dir": str(city_dir),
                "s2_path": str(s2_path),
                "s2_cloudmask_path": str(s2_cloudmask_path) if s2_cloudmask_path else "",
                "s2_diagnostics_path": str(s2_diagnostics_path) if s2_diagnostics_path else "",
                "s1_city_dir": str(s1_city_dir),
                "s1_path": str(s1_path) if s1_path else "",
                "has_s1": bool(s1_path),
                "final_label_vector": cfg["final_label_vector"],
                "original_polygon_vector": cfg.get("original_polygon_vector", ""),
            }
        )

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.sort_values("city")

    out_csv = metadata_dir / "city_input_inventory.csv"
    df.to_csv(out_csv, index=False)

    print(f"[OK] Inventory written to: {out_csv}")
    print(f"[INFO] Number of cities with S2: {len(df)}")

    if len(df) > 0:
        print(f"[INFO] Number of cities with S1: {int(df['has_s1'].sum())}")
        print()
        print("[PREVIEW]")
        print(df[["city", "s2_path", "has_s1", "s1_path"]].to_string(index=False))


if __name__ == "__main__":
    main()
