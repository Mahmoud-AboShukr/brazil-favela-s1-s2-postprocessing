from pathlib import Path
import argparse
import shutil
import yaml
import pandas as pd
import rasterio
from tqdm import tqdm


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def copy_geotiff(src, dst, compress=True, overwrite=False):
    src = Path(src)
    dst = Path(dst)

    if dst.exists() and not overwrite:
        return "SKIPPED_EXISTS"

    dst.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src) as ds:
        profile = ds.profile.copy()
        if compress:
            profile.update(
                compress="deflate",
                tiled=True,
                bigtiff="IF_SAFER"
            )

        data = ds.read()

        with rasterio.open(dst, "w", **profile) as out:
            out.write(data)
            out.update_tags(**ds.tags())
            for b in range(1, ds.count + 1):
                out.update_tags(b, **ds.tags(b))

    return "OK"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--inventory", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])
    inventory_path = Path(args.inventory) if args.inventory else output_root / "metadata" / "city_input_inventory.csv"

    df = pd.read_csv(inventory_path)

    s2_out_dir = output_root / "s2_final"
    cloud_out_dir = output_root / "auxiliary" / "cloud_masks"
    qc_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Finalizing S2"):
        city = row["city"]
        s2_src = Path(row["s2_path"])
        cloud_src = Path(row["s2_cloudmask_path"]) if isinstance(row["s2_cloudmask_path"], str) and row["s2_cloudmask_path"] else None

        s2_dst = s2_out_dir / city / f"{city}_s2_allbands_10m.tif"
        cloud_dst = cloud_out_dir / city / f"{city}_cloudmask.tif"

        status = "FAILED"
        error = ""

        try:
            status = copy_geotiff(
                s2_src,
                s2_dst,
                compress=cfg.get("compress", True),
                overwrite=cfg.get("overwrite", False),
            )

            cloud_status = "NO_CLOUDMASK"
            if cloud_src and cloud_src.exists():
                cloud_status = copy_geotiff(
                    cloud_src,
                    cloud_dst,
                    compress=cfg.get("compress", True),
                    overwrite=cfg.get("overwrite", False),
                )
            else:
                cloud_dst = ""

            with rasterio.open(s2_dst) as ds:
                qc_rows.append({
                    "city": city,
                    "s2_source": str(s2_src),
                    "s2_output": str(s2_dst),
                    "cloudmask_output": str(cloud_dst),
                    "status": status,
                    "cloud_status": cloud_status,
                    "band_count": ds.count,
                    "width": ds.width,
                    "height": ds.height,
                    "crs": str(ds.crs),
                    "resolution_x": ds.res[0],
                    "resolution_y": ds.res[1],
                    "dtype": ds.dtypes[0],
                    "error": error,
                })

        except Exception as e:
            qc_rows.append({
                "city": city,
                "s2_source": str(s2_src),
                "s2_output": str(s2_dst),
                "cloudmask_output": "",
                "status": "FAILED",
                "cloud_status": "FAILED",
                "band_count": "",
                "width": "",
                "height": "",
                "crs": "",
                "resolution_x": "",
                "resolution_y": "",
                "dtype": "",
                "error": repr(e),
            })

    qc_dir = output_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    out_csv = qc_dir / "s2_finalization_qc.csv"
    pd.DataFrame(qc_rows).to_csv(out_csv, index=False)

    print(f"[OK] S2 QC written to: {out_csv}")


if __name__ == "__main__":
    main()
