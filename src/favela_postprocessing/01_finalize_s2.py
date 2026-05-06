from pathlib import Path
import argparse
import yaml
import pandas as pd
import rasterio
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def copy_raster(src_path: Path, dst_path: Path, compress: bool = True, overwrite: bool = False):
    """
    Copy a raster while preserving all bands, CRS, transform, dtype, and tags.

    We force safe tiling options because some source rasters may have profiles
    that are not directly writable after modification.
    """
    if dst_path.exists() and not overwrite:
        return "SKIPPED_EXISTS"

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            count=src.count,
            height=src.height,
            width=src.width,
            crs=src.crs,
            transform=src.transform,
            dtype=src.dtypes[0],
            nodata=src.nodata,
            bigtiff="IF_SAFER",
        )

        if compress:
            profile.update(
                compress="deflate",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
        else:
            profile.update(
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                data = src.read(band_idx)
                dst.write(data, band_idx)
                dst.update_tags(band_idx, **src.tags(band_idx))

            dst.update_tags(**src.tags())

    return "OK"


def inspect_raster(path: Path):
    with rasterio.open(path) as src:
        return {
            "band_count": src.count,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "resolution_x": src.res[0],
            "resolution_y": src.res[1],
            "dtype": ",".join(src.dtypes),
            "nodata": src.nodata,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Finalize Sentinel-2 products while keeping all downloaded bands."
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

    s2_out_root = output_root / "s2_final"
    cloud_out_root = output_root / "auxiliary" / "cloud_masks"
    qc_dir = output_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    qc_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Finalizing S2"):
        city = row["city"]
        source_group = row.get("source_group", "")

        s2_src = Path(row["s2_path"])

        cloud_src_value = row.get("s2_cloudmask_path", "")
        cloud_src = None
        if isinstance(cloud_src_value, str) and cloud_src_value.strip():
            cloud_src = Path(cloud_src_value)

        s2_dst = s2_out_root / city / f"{city}_s2_allbands_10m.tif"
        cloud_dst = cloud_out_root / city / f"{city}_cloudmask.tif"

        row_qc = {
            "city": city,
            "source_group": source_group,
            "s2_source": str(s2_src),
            "s2_output": str(s2_dst),
            "cloudmask_source": str(cloud_src) if cloud_src else "",
            "cloudmask_output": str(cloud_dst) if cloud_src else "",
            "s2_status": "",
            "cloudmask_status": "",
            "band_count": "",
            "width": "",
            "height": "",
            "crs": "",
            "resolution_x": "",
            "resolution_y": "",
            "dtype": "",
            "nodata": "",
            "error": "",
        }

        try:
            if not s2_src.exists():
                raise FileNotFoundError(f"S2 source does not exist: {s2_src}")

            s2_status = copy_raster(
                src_path=s2_src,
                dst_path=s2_dst,
                compress=cfg.get("compress", True),
                overwrite=cfg.get("overwrite", False),
            )

            info = inspect_raster(s2_dst)

            row_qc.update(
                {
                    "s2_status": s2_status,
                    "band_count": info["band_count"],
                    "width": info["width"],
                    "height": info["height"],
                    "crs": info["crs"],
                    "resolution_x": info["resolution_x"],
                    "resolution_y": info["resolution_y"],
                    "dtype": info["dtype"],
                    "nodata": info["nodata"],
                }
            )

            if cloud_src and cloud_src.exists():
                cloud_status = copy_raster(
                    src_path=cloud_src,
                    dst_path=cloud_dst,
                    compress=cfg.get("compress", True),
                    overwrite=cfg.get("overwrite", False),
                )
            else:
                cloud_status = "NO_CLOUDMASK"

            row_qc["cloudmask_status"] = cloud_status

        except Exception as e:
            row_qc["s2_status"] = "FAILED"
            row_qc["cloudmask_status"] = "NOT_PROCESSED"
            row_qc["error"] = repr(e)

        qc_rows.append(row_qc)

    qc_df = pd.DataFrame(qc_rows)
    out_csv = qc_dir / "s2_finalization_qc.csv"
    qc_df.to_csv(out_csv, index=False)

    print(f"[OK] S2 finalization QC written to: {out_csv}")
    print()
    print("[SUMMARY]")
    print(qc_df["s2_status"].value_counts(dropna=False).to_string())
    print()
    print("[BAND COUNTS]")
    print(qc_df["band_count"].value_counts(dropna=False).to_string())
    print()
    print("[FAILED]")
    failed = qc_df[qc_df["s2_status"] == "FAILED"]
    if len(failed) == 0:
        print("None")
    else:
        print(failed[["city", "s2_source", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
