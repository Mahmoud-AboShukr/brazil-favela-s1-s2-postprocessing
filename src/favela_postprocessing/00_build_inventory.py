from pathlib import Path
import argparse
import yaml
import pandas as pd


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize_city_name(name: str) -> str:
    name = name.lower()
    for suffix in [
        "_v1", "_v2", "_v3", "_v4", "_v5", "_v6", "_v7",
        "_V1", "_V2", "_V3", "_V4", "_V5", "_V6", "_V7",
        "_ocm_v1", "_ocm_V1"
    ]:
        name = name.replace(suffix.lower(), "")
    return name.strip("_")


def find_s2_composite(city_dir: Path):
    preferred = [
        "city_composite_2022_smallholes_filled.tif",
        "city_composite_2022.tif",
    ]
    for fname in preferred:
        p = city_dir / fname
        if p.exists():
            return p

    matches = sorted(city_dir.rglob("*composite*.tif"))
    return matches[0] if matches else None


def find_s2_cloudmask(city_dir: Path):
    matches = sorted(city_dir.rglob("*cloudmask*.tif"))
    return matches[0] if matches else None


def find_s1_product(s1_city_dir: Path):
    if not s1_city_dir.exists():
        return None

    matches = sorted(s1_city_dir.rglob("*.tif"))
    if not matches:
        return None

    # Prefer files that sound aligned/final/composite-like if present
    priority_words = ["aligned", "rtc", "vv", "vh"]
    scored = []
    for p in matches:
        score = sum(w in p.name.lower() for w in priority_words)
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], str(x[1])))
    return scored[0][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    s2_root = Path(cfg["s2_root"])
    s1_root = Path(cfg["s1_root"])
    output_root = Path(cfg["output_root"])
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for city_dir in sorted(s2_root.iterdir()):
        if not city_dir.is_dir():
            continue

        s2_path = find_s2_composite(city_dir)
        if s2_path is None:
            continue

        city = normalize_city_name(city_dir.name)
        s1_path = find_s1_product(s1_root / city)
        cloudmask_path = find_s2_cloudmask(city_dir)

        rows.append({
            "city": city,
            "s2_city_dir": str(city_dir),
            "s2_path": str(s2_path),
            "s2_cloudmask_path": str(cloudmask_path) if cloudmask_path else "",
            "s1_city_dir": str(s1_root / city),
            "s1_path": str(s1_path) if s1_path else "",
            "has_s1": bool(s1_path),
            "final_label_vector": cfg["final_label_vector"],
            "original_polygon_vector": cfg.get("original_polygon_vector", ""),
        })

    df = pd.DataFrame(rows).sort_values("city")
    out_csv = metadata_dir / "city_input_inventory.csv"
    df.to_csv(out_csv, index=False)

    print(f"[OK] Inventory written to: {out_csv}")
    print(f"[INFO] Cities with S2: {len(df)}")
    print(f"[INFO] Cities with S1: {int(df['has_s1'].sum())}")


if __name__ == "__main__":
    main()
