from pathlib import Path
import argparse
import yaml
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from tqdm import tqdm


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def rasterize_vector_to_reference(vector_gdf, reference_tif, output_tif, compress=True, overwrite=False):
    output_tif = Path(output_tif)

    if output_tif.exists() and not overwrite:
        return "SKIPPED_EXISTS", 0

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_tif) as ref:
        ref_bounds_geom = box(*ref.bounds)
        ref_crs = ref.crs
        out_shape = (ref.height, ref.width)
        transform = ref.transform

        gdf = vector_gdf
        if gdf.crs != ref_crs:
            gdf = gdf.to_crs(ref_crs)

        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[gdf.geometry.intersects(ref_bounds_geom)]

        n_features = len(gdf)

        shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)

        mask = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )

        profile = ref.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate" if compress else None,
            tiled=True,
            bigtiff="IF_SAFER"
        )

        with rasterio.open(output_tif, "w", **profile) as out:
            out.write(mask, 1)
            out.update_tags(
                source_vector="final_label_vector",
                reference_grid=str(reference_tif),
                label_value_0="background",
                label_value_1="favela"
            )

    return "OK", n_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--inventory", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])
    inventory_path = Path(args.inventory) if args.inventory else output_root / "metadata" / "city_input_inventory.csv"

    df = pd.read_csv(inventory_path)

    final_label_vector = Path(cfg["final_label_vector"])
    original_polygon_vector = Path(cfg["original_polygon_vector"]) if cfg.get("original_polygon_vector") else None

    print(f"[LOAD] Final label vector: {final_label_vector}")
    final_gdf = gpd.read_file(final_label_vector)

    original_gdf = None
    if original_polygon_vector and original_polygon_vector.exists():
        print(f"[LOAD] Original polygon vector: {original_polygon_vector}")
        original_gdf = gpd.read_file(original_polygon_vector)

    label_out_dir = output_root / "labels_final"
    original_out_dir = output_root / "auxiliary" / "original_polygon_masks"

    qc_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Finalizing labels"):
        city = row["city"]
        s2_ref = output_root / "s2_final" / city / f"{city}_s2_allbands_10m.tif"
        label_dst = label_out_dir / city / f"{city}_label_final.tif"
        original_dst = original_out_dir / city / f"{city}_mask_original_polygon.tif"

        if not s2_ref.exists():
            qc_rows.append({
                "city": city,
                "reference_s2": str(s2_ref),
                "label_output": str(label_dst),
                "status": "MISSING_S2_REFERENCE",
                "n_final_features": "",
                "n_original_features": "",
                "error": "",
            })
            continue

        try:
            status, n_final = rasterize_vector_to_reference(
                final_gdf,
                s2_ref,
                label_dst,
                compress=cfg.get("compress", True),
                overwrite=cfg.get("overwrite", False),
            )

            n_original = ""
            original_status = "NOT_CREATED"
            if original_gdf is not None:
                original_status, n_original = rasterize_vector_to_reference(
                    original_gdf,
                    s2_ref,
                    original_dst,
                    compress=cfg.get("compress", True),
                    overwrite=cfg.get("overwrite", False),
                )

            with rasterio.open(label_dst) as ds:
                arr = ds.read(1)
                positive_pixels = int((arr == 1).sum())
                total_pixels = int(arr.size)
                positive_percent = positive_pixels / total_pixels * 100.0 if total_pixels else 0.0

            qc_rows.append({
                "city": city,
                "reference_s2": str(s2_ref),
                "label_output": str(label_dst),
                "original_polygon_mask_output": str(original_dst) if original_gdf is not None else "",
                "status": status,
                "original_status": original_status,
                "n_final_features": n_final,
                "n_original_features": n_original,
                "positive_pixels": positive_pixels,
                "positive_percent": positive_percent,
                "error": "",
            })

        except Exception as e:
            qc_rows.append({
                "city": city,
                "reference_s2": str(s2_ref),
                "label_output": str(label_dst),
                "original_polygon_mask_output": "",
                "status": "FAILED",
                "original_status": "FAILED",
                "n_final_features": "",
                "n_original_features": "",
                "positive_pixels": "",
                "positive_percent": "",
                "error": repr(e),
            })

    qc_dir = output_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    out_csv = qc_dir / "label_finalization_qc.csv"
    pd.DataFrame(qc_rows).to_csv(out_csv, index=False)

    print(f"[OK] Label QC written to: {out_csv}")


if __name__ == "__main__":
    main()
