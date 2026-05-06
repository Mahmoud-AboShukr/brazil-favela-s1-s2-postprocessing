from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_db_auto(arr):
    arr = arr.astype("float32")
    valid = np.isfinite(arr)
    if not valid.any():
        return arr

    sample = arr[valid]
    p01 = np.nanpercentile(sample, 1)
    p99 = np.nanpercentile(sample, 99)

    # If values are non-negative and look linear, convert to dB.
    # If values are already negative/positive dB-like, keep them.
    if p01 >= 0 and p99 > 1:
        out = np.full_like(arr, np.nan, dtype="float32")
        positive = arr > 0
        out[positive] = 10.0 * np.log10(arr[positive])
        return out

    return arr


def align_s1_to_s2(s1_path, s2_path, out_path, create_vv_minus_vh=True, db_mode="auto", compress=True, overwrite=False):
    s1_path = Path(s1_path)
    s2_path = Path(s2_path)
    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        return "SKIPPED_EXISTS"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(s2_path) as s2, rasterio.open(s1_path) as s1:
        target_profile = s2.profile.copy()
        target_profile.update(
            count=3 if create_vv_minus_vh else min(2, s1.count),
            dtype="float32",
            nodata=np.nan,
            compress="deflate" if compress else None,
            tiled=True,
            bigtiff="IF_SAFER"
        )

        target_h, target_w = s2.height, s2.width
        aligned_bands = []

        n_input = min(2, s1.count)
        if n_input < 2:
            raise ValueError(f"S1 file must contain at least two bands for VV/VH. Found {s1.count}: {s1_path}")

        for band_idx in range(1, n_input + 1):
            src_arr = s1.read(band_idx).astype("float32")
            dst_arr = np.full((target_h, target_w), np.nan, dtype="float32")

            reproject(
                source=src_arr,
                destination=dst_arr,
                src_transform=s1.transform,
                src_crs=s1.crs,
                dst_transform=s2.transform,
                dst_crs=s2.crs,
                src_nodata=s1.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

            if db_mode == "auto":
                dst_arr = to_db_auto(dst_arr)
            elif db_mode == "linear_to_db":
                positive = dst_arr > 0
                converted = np.full_like(dst_arr, np.nan, dtype="float32")
                converted[positive] = 10.0 * np.log10(dst_arr[positive])
                dst_arr = converted
            elif db_mode == "already_db":
                pass
            else:
                raise ValueError(f"Unknown db_mode: {db_mode}")

            aligned_bands.append(dst_arr)

        if create_vv_minus_vh:
            vv = aligned_bands[0]
            vh = aligned_bands[1]
            diff = vv - vh
            aligned_bands.append(diff.astype("float32"))

        stack = np.stack(aligned_bands, axis=0)

        with rasterio.open(out_path, "w", **target_profile) as out:
            out.write(stack)
            out.update_tags(
                source_s1=str(s1_path),
                reference_s2=str(s2_path),
                band_1="VV_dB_or_original",
                band_2="VH_dB_or_original",
                band_3="VV_minus_VH" if create_vv_minus_vh else "not_created",
            )

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
    out_dir = output_root / "s1_final"
    qc_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Finalizing S1"):
        city = row["city"]
        s1_src = row.get("s1_path", "")

        s2_final = output_root / "s2_final" / city / f"{city}_s2_allbands_10m.tif"
        s1_dst = out_dir / city / f"{city}_s1_vv_vh_vvdiff_10m_aligned.tif"

        if not isinstance(s1_src, str) or not s1_src or not Path(s1_src).exists():
            qc_rows.append({
                "city": city,
                "s1_source": s1_src,
                "s1_output": str(s1_dst),
                "reference_s2": str(s2_final),
                "status": "MISSING_S1",
                "band_count": "",
                "width": "",
                "height": "",
                "crs": "",
                "error": "",
            })
            continue

        try:
            status = align_s1_to_s2(
                s1_src,
                s2_final,
                s1_dst,
                create_vv_minus_vh=cfg.get("s1_create_vv_minus_vh", True),
                db_mode=cfg.get("s1_db_mode", "auto"),
                compress=cfg.get("compress", True),
                overwrite=cfg.get("overwrite", False),
            )

            with rasterio.open(s1_dst) as ds:
                qc_rows.append({
                    "city": city,
                    "s1_source": s1_src,
                    "s1_output": str(s1_dst),
                    "reference_s2": str(s2_final),
                    "status": status,
                    "band_count": ds.count,
                    "width": ds.width,
                    "height": ds.height,
                    "crs": str(ds.crs),
                    "error": "",
                })

        except Exception as e:
            qc_rows.append({
                "city": city,
                "s1_source": s1_src,
                "s1_output": str(s1_dst),
                "reference_s2": str(s2_final),
                "status": "FAILED",
                "band_count": "",
                "width": "",
                "height": "",
                "crs": "",
                "error": repr(e),
            })

    qc_dir = output_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    out_csv = qc_dir / "s1_finalization_qc.csv"
    pd.DataFrame(qc_rows).to_csv(out_csv, index=False)

    print(f"[OK] S1 QC written to: {out_csv}")


if __name__ == "__main__":
    main()
