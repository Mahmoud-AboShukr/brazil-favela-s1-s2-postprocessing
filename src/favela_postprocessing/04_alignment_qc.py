from pathlib import Path
import argparse
import yaml

import pandas as pd
import rasterio


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def raster_info(path: Path):
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "exists": True,
            "crs": str(src.crs),
            "transform": tuple(round(v, 12) for v in src.transform),
            "width": src.width,
            "height": src.height,
            "res_x": round(src.res[0], 12),
            "res_y": round(src.res[1], 12),
            "count": src.count,
            "dtype": ",".join(src.dtypes),
            "nodata": src.nodata,
        }


def missing_info(path: Path):
    return {
        "path": str(path),
        "exists": False,
        "crs": "",
        "transform": "",
        "width": "",
        "height": "",
        "res_x": "",
        "res_y": "",
        "count": "",
        "dtype": "",
        "nodata": "",
    }


def compare_to_s2(s2, other):
    if not other["exists"]:
        return {
            "same_crs": False,
            "same_transform": False,
            "same_shape": False,
            "same_resolution": False,
            "all_match": False,
        }

    same_crs = s2["crs"] == other["crs"]
    same_transform = s2["transform"] == other["transform"]
    same_shape = (s2["width"] == other["width"]) and (s2["height"] == other["height"])
    same_resolution = (s2["res_x"] == other["res_x"]) and (s2["res_y"] == other["res_y"])

    return {
        "same_crs": same_crs,
        "same_transform": same_transform,
        "same_shape": same_shape,
        "same_resolution": same_resolution,
        "all_match": same_crs and same_transform and same_shape and same_resolution,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check alignment between finalized S2, S1, and label rasters."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--inventory", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])

    inventory_path = (
        Path(args.inventory)
        if args.inventory
        else output_root / "metadata" / "city_input_inventory.csv"
    )

    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")

    df = pd.read_csv(inventory_path)

    qc_rows = []

    for _, row in df.iterrows():
        city = row["city"]

        s2_path = output_root / "s2_final" / city / f"{city}_s2_allbands_10m.tif"
        s1_path = output_root / "s1_final" / city / f"{city}_s1_grd_vv_vh_vvdiff_10m_aligned.tif"
        label_path = output_root / "labels_final" / city / f"{city}_label_final.tif"
        original_mask_path = (
            output_root
            / "auxiliary"
            / "original_polygon_masks"
            / city
            / f"{city}_mask_original_polygon.tif"
        )

        row_qc = {
            "city": city,
            "s2_path": str(s2_path),
            "s1_path": str(s1_path),
            "label_path": str(label_path),
            "original_mask_path": str(original_mask_path),
            "s2_exists": s2_path.exists(),
            "s1_exists": s1_path.exists(),
            "label_exists": label_path.exists(),
            "original_mask_exists": original_mask_path.exists(),
            "s2_band_count": "",
            "s1_band_count": "",
            "label_band_count": "",
            "original_mask_band_count": "",
            "s2_crs": "",
            "s2_width": "",
            "s2_height": "",
            "s2_res_x": "",
            "s2_res_y": "",
            "s1_same_crs": "",
            "s1_same_transform": "",
            "s1_same_shape": "",
            "s1_same_resolution": "",
            "s1_alignment_ok": "",
            "label_same_crs": "",
            "label_same_transform": "",
            "label_same_shape": "",
            "label_same_resolution": "",
            "label_alignment_ok": "",
            "original_same_crs": "",
            "original_same_transform": "",
            "original_same_shape": "",
            "original_same_resolution": "",
            "original_alignment_ok": "",
            "city_status": "",
            "error": "",
        }

        try:
            if not s2_path.exists():
                row_qc["city_status"] = "FAILED_MISSING_S2"
                qc_rows.append(row_qc)
                continue

            s2 = raster_info(s2_path)
            s1 = raster_info(s1_path) if s1_path.exists() else missing_info(s1_path)
            label = raster_info(label_path) if label_path.exists() else missing_info(label_path)
            original = (
                raster_info(original_mask_path)
                if original_mask_path.exists()
                else missing_info(original_mask_path)
            )

            s1_cmp = compare_to_s2(s2, s1)
            label_cmp = compare_to_s2(s2, label)
            original_cmp = compare_to_s2(s2, original)

            row_qc.update(
                {
                    "s2_band_count": s2["count"],
                    "s1_band_count": s1["count"],
                    "label_band_count": label["count"],
                    "original_mask_band_count": original["count"],
                    "s2_crs": s2["crs"],
                    "s2_width": s2["width"],
                    "s2_height": s2["height"],
                    "s2_res_x": s2["res_x"],
                    "s2_res_y": s2["res_y"],

                    "s1_same_crs": s1_cmp["same_crs"],
                    "s1_same_transform": s1_cmp["same_transform"],
                    "s1_same_shape": s1_cmp["same_shape"],
                    "s1_same_resolution": s1_cmp["same_resolution"],
                    "s1_alignment_ok": s1_cmp["all_match"],

                    "label_same_crs": label_cmp["same_crs"],
                    "label_same_transform": label_cmp["same_transform"],
                    "label_same_shape": label_cmp["same_shape"],
                    "label_same_resolution": label_cmp["same_resolution"],
                    "label_alignment_ok": label_cmp["all_match"],

                    "original_same_crs": original_cmp["same_crs"],
                    "original_same_transform": original_cmp["same_transform"],
                    "original_same_shape": original_cmp["same_shape"],
                    "original_same_resolution": original_cmp["same_resolution"],
                    "original_alignment_ok": original_cmp["all_match"],
                }
            )

            if label_cmp["all_match"] and s1_cmp["all_match"]:
                row_qc["city_status"] = "OK_S2_S1_LABEL"
            elif label_cmp["all_match"] and not s1_path.exists():
                row_qc["city_status"] = "OK_S2_LABEL_S1_MISSING"
            elif label_cmp["all_match"] and not s1_cmp["all_match"]:
                row_qc["city_status"] = "LABEL_OK_S1_ALIGNMENT_PROBLEM"
            else:
                row_qc["city_status"] = "ALIGNMENT_PROBLEM"

        except Exception as e:
            row_qc["city_status"] = "FAILED_EXCEPTION"
            row_qc["error"] = repr(e)

        qc_rows.append(row_qc)

    qc_df = pd.DataFrame(qc_rows)

    qc_dir = output_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    out_csv = qc_dir / "alignment_qc.csv"
    qc_df.to_csv(out_csv, index=False)

    print(f"[OK] Alignment QC written to: {out_csv}")
    print()

    print("[CITY STATUS]")
    print(qc_df["city_status"].value_counts(dropna=False).to_string())
    print()

    print("[S2 + LABEL ALIGNMENT]")
    print(qc_df["label_alignment_ok"].value_counts(dropna=False).to_string())
    print()

    print("[S2 + S1 ALIGNMENT]")
    print(qc_df["s1_alignment_ok"].value_counts(dropna=False).to_string())
    print()

    print("[FAILED / PROBLEMATIC ROWS]")
    problematic = qc_df[
        ~qc_df["city_status"].isin(["OK_S2_S1_LABEL", "OK_S2_LABEL_S1_MISSING"])
    ]

    if len(problematic) == 0:
        print("None")
    else:
        cols = [
            "city",
            "city_status",
            "s2_exists",
            "s1_exists",
            "label_exists",
            "s1_alignment_ok",
            "label_alignment_ok",
            "error",
        ]
        print(problematic[cols].to_string(index=False))

    print()
    print("[S1 MISSING ROWS]")
    missing_s1 = qc_df[qc_df["s1_exists"] == False]
    if len(missing_s1) == 0:
        print("None")
    else:
        print(missing_s1[["city", "city_status", "s1_path"]].to_string(index=False))


if __name__ == "__main__":
    main()
