from pathlib import Path
import argparse
import yaml
import pandas as pd


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize_city_name(name: str) -> str:
    name = str(name).lower().strip()

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
    preferred_names = [
        "city_composite_2022_smallholes_filled.tif",
        "city_composite_2022.tif",
    ]

    for name in preferred_names:
        candidate = city_dir / name
        if candidate.exists():
            return candidate

    matches = sorted(city_dir.rglob("*smallholes_filled*.tif"))
    if matches:
        return matches[0]

    matches = sorted(city_dir.rglob("*composite*.tif"))
    if matches:
        filtered = [
            p for p in matches
            if "cloudmask" not in p.name.lower()
            and "observation" not in p.name.lower()
            and "small_holes_mask" not in p.name.lower()
        ]
        return filtered[0] if filtered else matches[0]

    return None


def find_s2_cloudmask(city_dir: Path):
    matches = sorted(city_dir.rglob("*cloudmask*.tif"))
    return matches[0] if matches else None


def find_s2_diagnostics(city_dir: Path):
    matches = sorted(city_dir.rglob("*diagnostics*.json"))
    return matches[0] if matches else None


def deduplicate_paths(paths):
    seen = set()
    out = []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def find_vv_vh_paths(s1_city_dir: Path):
    """
    Find separate Sentinel-1 GRD VV.tif and VH.tif files for a city.
    Expected structure:
    raw/<city>/<S1_ITEM_ID>/VV.tif
    raw/<city>/<S1_ITEM_ID>/VH.tif
    """
    if not s1_city_dir.exists():
        return None, None

    vv_candidates = []
    vh_candidates = []

    vv_candidates.extend(sorted(s1_city_dir.rglob("VV.tif")))
    vv_candidates.extend(sorted(s1_city_dir.rglob("*VV*.tif")))
    vv_candidates.extend(sorted(s1_city_dir.rglob("*vv*.tif")))

    vh_candidates.extend(sorted(s1_city_dir.rglob("VH.tif")))
    vh_candidates.extend(sorted(s1_city_dir.rglob("*VH*.tif")))
    vh_candidates.extend(sorted(s1_city_dir.rglob("*vh*.tif")))

    vv_candidates = deduplicate_paths(vv_candidates)
    vh_candidates = deduplicate_paths(vh_candidates)

    vv_path = vv_candidates[0] if vv_candidates else None
    vh_path = vh_candidates[0] if vh_candidates else None

    return vv_path, vh_path


def add_city_row(rows, city_name, city_dir, s1_root, cfg, source_group):
    s2_path = find_s2_composite(city_dir)
    if s2_path is None:
        return

    city = normalize_city_name(city_name)

    s2_cloudmask_path = find_s2_cloudmask(city_dir)
    s2_diagnostics_path = find_s2_diagnostics(city_dir)

    s1_city_dir = s1_root / city
    vv_path, vh_path = find_vv_vh_paths(s1_city_dir)

    rows.append(
        {
            "city": city,
            "source_group": source_group,
            "s2_folder_name": city_dir.name,
            "s2_city_dir": str(city_dir),
            "s2_path": str(s2_path),
            "s2_cloudmask_path": str(s2_cloudmask_path) if s2_cloudmask_path else "",
            "s2_diagnostics_path": str(s2_diagnostics_path) if s2_diagnostics_path else "",
            "s1_city_dir": str(s1_city_dir),
            "s1_vv_path": str(vv_path) if vv_path else "",
            "s1_vh_path": str(vh_path) if vh_path else "",
            "has_s1": bool(vv_path and vh_path),
            "final_label_vector": cfg["final_label_vector"],
            "original_polygon_vector": cfg.get("original_polygon_vector", ""),
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build city-level input inventory for S1/S2/favela post-processing."
    )
    parser.add_argument("--config", default="configs/default.yaml")
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

        if city_dir.name.lower() == "ocm_v1":
            for ocm_city_dir in sorted(city_dir.iterdir()):
                if not ocm_city_dir.is_dir():
                    continue

                add_city_row(
                    rows=rows,
                    city_name=ocm_city_dir.name,
                    city_dir=ocm_city_dir,
                    s1_root=s1_root,
                    cfg=cfg,
                    source_group="ocm_v1",
                )
            continue

        add_city_row(
            rows=rows,
            city_name=city_dir.name,
            city_dir=city_dir,
            s1_root=s1_root,
            cfg=cfg,
            source_group="standard",
        )

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df = df.drop_duplicates(subset=["city"], keep="last")
        df = df.sort_values("city")

    out_csv = metadata_dir / "city_input_inventory.csv"
    df.to_csv(out_csv, index=False)

    print(f"[OK] Inventory written to: {out_csv}")
    print(f"[INFO] Number of cities with S2: {len(df)}")

    if len(df) > 0:
        print(f"[INFO] Number of cities with complete S1 VV/VH: {int(df['has_s1'].sum())}")
        print()
        print("[MISSING S1 VV/VH]")
        missing = df[df["has_s1"] == False]["city"].tolist()
        print(missing if missing else "None")
        print()
        print("[PREVIEW]")
        print(
            df[
                ["city", "source_group", "has_s1", "s1_vv_path", "s1_vh_path"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
