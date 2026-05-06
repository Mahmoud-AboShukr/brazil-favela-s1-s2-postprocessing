from pathlib import Path
import argparse
import yaml

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Basic geometry cleaning:
    - remove null/empty geometries
    - attempt to repair invalid geometries
    """
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    try:
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
    except Exception:
        pass

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    return gdf


def rasterize_vector_to_reference(
    gdf: gpd.GeoDataFrame,
    reference_tif: Path,
    output_tif: Path,
    source_name: str,
    compress: bool = True,
    overwrite: bool = False,
    all_touched: bool = False,
):
    """
    Rasterize a vector layer onto the exact grid of a reference raster.

    Output:
    - uint8 raster
    - 0 = background
    - 1 = favela
    """
    if output_tif.exists() and not overwrite:
        with rasterio.open(output_tif) as src:
            arr = src.read(1)
            positive_pixels = int((arr == 1).sum())
            total_pixels = int(arr.size)
            positive_percent = (
                positive_pixels / total_pixels * 100.0 if total_pixels else 0.0
            )

        return "SKIPPED_EXISTS", None, positive_pixels, positive_percent

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_tif) as ref:
        ref_crs = ref.crs
        ref_bounds = ref.bounds
        ref_bounds_geom = box(
            ref_bounds.left,
            ref_bounds.bottom,
            ref_bounds.right,
            ref_bounds.top,
        )

        out_shape = (ref.height, ref.width)
        transform = ref.transform

        local_gdf = gdf

        if local_gdf.crs != ref_crs:
            local_gdf = local_gdf.to_crs(ref_crs)

        local_gdf = clean_geometries(local_gdf)

        # Keep only features intersecting the city raster bounds.
        local_gdf = local_gdf[local_gdf.geometry.intersects(ref_bounds_geom)].copy()
        n_features = len(local_gdf)

        if n_features > 0:
            shapes = (
                (geom, 1)
                for geom in local_gdf.geometry
                if geom is not None and not geom.is_empty
            )
        else:
            shapes = []

        mask = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        )

        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate" if compress else None,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            bigtiff="IF_SAFER",
        )

        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(mask, 1)
            dst.update_tags(
                source_vector=source_name,
                reference_grid=str(reference_tif),
                label_value_0="background",
                label_value_1="favela",
                all_touched=str(all_touched),
            )
            dst.set_band_description(1, "favela_binary_label")

        positive_pixels = int((mask == 1).sum())
        total_pixels = int(mask.size)
        positive_percent = (
            positive_pixels / total_pixels * 100.0 if total_pixels else 0.0
        )

    return "OK", n_features, positive_pixels, positive_percent


def main():
    parser = argparse.ArgumentParser(
        description="Rasterize final favela labels to the finalized Sentinel-2 grid."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--inventory", default=None)
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Rasterize polygons using all_touched=True. Default is False.",
    )
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

    final_label_vector = Path(cfg["final_label_vector"])
    original_polygon_vector = (
        Path(cfg["original_polygon_vector"])
        if cfg.get("original_polygon_vector")
        else None
    )

    if not final_label_vector.exists():
        raise FileNotFoundError(f"Final label vector not found: {final_label_vector}")

    print(f"[LOAD] Final label vector: {final_label_vector}")
    final_gdf = gpd.read_file(final_label_vector)
    final_gdf = clean_geometries(final_gdf)
    print(f"[INFO] Final label features loaded: {len(final_gdf)}")
    print(f"[INFO] Final label CRS: {final_gdf.crs}")

    original_gdf = None
    if original_polygon_vector and original_polygon_vector.exists():
        print(f"[LOAD] Original polygon vector: {original_polygon_vector}")
        original_gdf = gpd.read_file(original_polygon_vector)
        original_gdf = clean_geometries(original_gdf)
        print(f"[INFO] Original polygon features loaded: {len(original_gdf)}")
        print(f"[INFO] Original polygon CRS: {original_gdf.crs}")
    else:
        print("[INFO] Original polygon vector not found or not configured. Skipping auxiliary original masks.")

    df = pd.read_csv(inventory_path)

    labels_out_root = output_root / "labels_final"
    original_out_root = output_root / "auxiliary" / "original_polygon_masks"
    qc_dir = output_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    qc_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Finalizing labels"):
        city = row["city"]

        s2_ref = output_root / "s2_final" / city / f"{city}_s2_allbands_10m.tif"
        label_out = labels_out_root / city / f"{city}_label_final.tif"
        original_out = original_out_root / city / f"{city}_mask_original_polygon.tif"

        row_qc = {
            "city": city,
            "reference_s2": str(s2_ref),
            "label_output": str(label_out),
            "original_polygon_mask_output": str(original_out) if original_gdf is not None else "",
            "label_status": "",
            "original_status": "",
            "n_final_features_intersecting": "",
            "n_original_features_intersecting": "",
            "label_positive_pixels": "",
            "label_positive_percent": "",
            "original_positive_pixels": "",
            "original_positive_percent": "",
            "error": "",
        }

        try:
            if not s2_ref.exists():
                raise FileNotFoundError(
                    f"Missing finalized S2 reference for {city}: {s2_ref}"
                )

            (
                label_status,
                n_final,
                label_positive_pixels,
                label_positive_percent,
            ) = rasterize_vector_to_reference(
                gdf=final_gdf,
                reference_tif=s2_ref,
                output_tif=label_out,
                source_name=str(final_label_vector),
                compress=cfg.get("compress", True),
                overwrite=cfg.get("overwrite", False),
                all_touched=args.all_touched,
            )

            row_qc.update(
                {
                    "label_status": label_status,
                    "n_final_features_intersecting": n_final,
                    "label_positive_pixels": label_positive_pixels,
                    "label_positive_percent": label_positive_percent,
                }
            )

            if original_gdf is not None:
                (
                    original_status,
                    n_original,
                    original_positive_pixels,
                    original_positive_percent,
                ) = rasterize_vector_to_reference(
                    gdf=original_gdf,
                    reference_tif=s2_ref,
                    output_tif=original_out,
                    source_name=str(original_polygon_vector),
                    compress=cfg.get("compress", True),
                    overwrite=cfg.get("overwrite", False),
                    all_touched=args.all_touched,
                )

                row_qc.update(
                    {
                        "original_status": original_status,
                        "n_original_features_intersecting": n_original,
                        "original_positive_pixels": original_positive_pixels,
                        "original_positive_percent": original_positive_percent,
                    }
                )
            else:
                row_qc["original_status"] = "NOT_CREATED"

        except Exception as e:
            row_qc["label_status"] = "FAILED"
            row_qc["original_status"] = "FAILED"
            row_qc["error"] = repr(e)

        qc_rows.append(row_qc)

    qc_df = pd.DataFrame(qc_rows)
    out_csv = qc_dir / "label_finalization_qc.csv"
    qc_df.to_csv(out_csv, index=False)

    print(f"[OK] Label finalization QC written to: {out_csv}")
    print()
    print("[SUMMARY]")
    print(qc_df["label_status"].value_counts(dropna=False).to_string())
    print()
    print("[ORIGINAL MASK SUMMARY]")
    print(qc_df["original_status"].value_counts(dropna=False).to_string())
    print()
    print("[LABEL POSITIVE PERCENT SUMMARY]")
    print(qc_df["label_positive_percent"].describe().to_string())
    print()
    print("[FAILED]")
    failed = qc_df[qc_df["label_status"] == "FAILED"]
    if len(failed) == 0:
        print("None")
    else:
        print(failed[["city", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
