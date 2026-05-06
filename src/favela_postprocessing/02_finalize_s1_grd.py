from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm


OUT_NODATA = -9999.0


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_db_auto(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")
    valid = np.isfinite(arr) & (arr != OUT_NODATA)

    if not valid.any():
        return arr

    sample = arr[valid]
    p01 = np.nanpercentile(sample, 1)
    p99 = np.nanpercentile(sample, 99)

    # If values look linear and positive, convert to dB.
    # If values already look dB-like, keep them.
    if p01 >= 0 and p99 > 1:
        out = np.full_like(arr, OUT_NODATA, dtype="float32")
        positive = valid & (arr > 0)
        out[positive] = 10.0 * np.log10(arr[positive])
        return out

    return arr.astype("float32")


def read_reproject_single_band(src_path: Path, ref, db_mode: str) -> tuple[np.ndarray, str]:
    """
    Reproject one S1 GRD band to the finalized S2 grid.

    Some Sentinel-1 GRD rasters from Planetary Computer have no normal affine
    transform/CRS. They may expose GCPs instead. Rasterio.reproject expects the
    keyword argument `gcps=...`, not `src_gcps=...`.
    """
    dst_arr = np.full((ref.height, ref.width), OUT_NODATA, dtype="float32")

    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype("float32")

        if src.crs is not None:
            georef_mode = "affine_crs"
            reproject(
                source=src_arr,
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=OUT_NODATA,
                resampling=Resampling.bilinear,
            )

        else:
            gcps, gcp_crs = src.gcps

            if not gcps or gcp_crs is None:
                raise ValueError(
                    f"{src_path} has neither src.crs nor usable GCPs. "
                    "This raw S1 file cannot be aligned directly with Rasterio."
                )

            georef_mode = "gcps"
            reproject(
                source=src_arr,
                destination=dst_arr,
                gcps=gcps,
                src_crs=gcp_crs,
                src_nodata=src.nodata,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=OUT_NODATA,
                resampling=Resampling.bilinear,
            )

    if db_mode == "auto":
        dst_arr = to_db_auto(dst_arr)
    elif db_mode == "linear_to_db":
        out = np.full_like(dst_arr, OUT_NODATA, dtype="float32")
        valid = np.isfinite(dst_arr) & (dst_arr != OUT_NODATA) & (dst_arr > 0)
        out[valid] = 10.0 * np.log10(dst_arr[valid])
        dst_arr = out
    elif db_mode == "already_db":
        dst_arr = dst_arr.astype("float32")
    else:
        raise ValueError(f"Unknown db_mode: {db_mode}")

    return dst_arr, georef_mode


def finalize_s1_city(
    vv_path: Path,
    vh_path: Path,
    s2_ref_path: Path,
    out_path: Path,
    cfg: dict,
):
    overwrite = cfg.get("overwrite", False)
    compress = cfg.get("compress", True)
    db_mode = cfg.get("s1_db_mode", "auto")
    create_diff = cfg.get("s1_create_vv_minus_vh", True)

    if out_path.exists() and not overwrite:
        return "SKIPPED_EXISTS", "", "", None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(s2_ref_path) as ref:
        vv, vv_georef_mode = read_reproject_single_band(vv_path, ref, db_mode=db_mode)
        vh, vh_georef_mode = read_reproject_single_band(vh_path, ref, db_mode=db_mode)

        bands = [vv, vh]
        band_names = ["VV_dB", "VH_dB"]

        if create_diff:
            diff = np.full_like(vv, OUT_NODATA, dtype="float32")
            valid = (
                np.isfinite(vv)
                & np.isfinite(vh)
                & (vv != OUT_NODATA)
                & (vh != OUT_NODATA)
            )
            diff[valid] = vv[valid] - vh[valid]
            bands.append(diff)
            band_names.append("VV_minus_VH_dB")

        stack = np.stack(bands, axis=0).astype("float32")

        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            count=stack.shape[0],
            dtype="float32",
            nodata=OUT_NODATA,
            compress="deflate" if compress else None,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            bigtiff="IF_SAFER",
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(stack)
            dst.update_tags(
                source_vv=str(vv_path),
                source_vh=str(vh_path),
                reference_s2=str(s2_ref_path),
                product="Sentinel-1 GRD aligned to Sentinel-2 grid",
                db_mode=db_mode,
                vv_georef_mode=vv_georef_mode,
                vh_georef_mode=vh_georef_mode,
                band_order=",".join(band_names),
                nodata=str(OUT_NODATA),
            )

            for idx, name in enumerate(band_names, start=1):
                dst.update_tags(idx, band_name=name)

    valid_pixels = np.isfinite(stack) & (stack != OUT_NODATA)
    valid_fraction = float(valid_pixels.all(axis=0).mean())

    return "OK", vv_georef_mode, vh_georef_mode, valid_fraction


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
        description="Finalize Sentinel-1 GRD VV/VH and align to finalized Sentinel-2 grid."
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

    s1_out_root = output_root / "s1_final"
    qc_dir = output_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    qc_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Finalizing S1 GRD"):
        city = row["city"]

        vv_value = row.get("s1_vv_path", "")
        vh_value = row.get("s1_vh_path", "")

        vv_path = Path(vv_value) if isinstance(vv_value, str) and vv_value.strip() else None
        vh_path = Path(vh_value) if isinstance(vh_value, str) and vh_value.strip() else None

        s2_ref = output_root / "s2_final" / city / f"{city}_s2_allbands_10m.tif"
        s1_out = s1_out_root / city / f"{city}_s1_grd_vv_vh_vvdiff_10m_aligned.tif"

        row_qc = {
            "city": city,
            "s1_vv_source": str(vv_path) if vv_path else "",
            "s1_vh_source": str(vh_path) if vh_path else "",
            "reference_s2": str(s2_ref),
            "s1_output": str(s1_out),
            "status": "",
            "band_count": "",
            "width": "",
            "height": "",
            "crs": "",
            "resolution_x": "",
            "resolution_y": "",
            "dtype": "",
            "vv_georef_mode": "",
            "vh_georef_mode": "",
            "valid_fraction_all_bands": "",
            "error": "",
        }

        try:
            if vv_path is None or not vv_path.exists():
                raise FileNotFoundError(f"Missing VV file for {city}: {vv_path}")

            if vh_path is None or not vh_path.exists():
                raise FileNotFoundError(f"Missing VH file for {city}: {vh_path}")

            if not s2_ref.exists():
                raise FileNotFoundError(f"Missing finalized S2 reference for {city}: {s2_ref}")

            status, vv_georef_mode, vh_georef_mode, valid_fraction = finalize_s1_city(
                vv_path=vv_path,
                vh_path=vh_path,
                s2_ref_path=s2_ref,
                out_path=s1_out,
                cfg=cfg,
            )

            info = inspect_raster(s1_out)

            row_qc.update(
                {
                    "status": status,
                    "band_count": info["band_count"],
                    "width": info["width"],
                    "height": info["height"],
                    "crs": info["crs"],
                    "resolution_x": info["resolution_x"],
                    "resolution_y": info["resolution_y"],
                    "dtype": info["dtype"],
                    "vv_georef_mode": vv_georef_mode,
                    "vh_georef_mode": vh_georef_mode,
                    "valid_fraction_all_bands": valid_fraction,
                }
            )

        except Exception as e:
            row_qc["status"] = "FAILED"
            row_qc["error"] = repr(e)

        qc_rows.append(row_qc)

    qc_df = pd.DataFrame(qc_rows)
    out_csv = qc_dir / "s1_grd_finalization_qc.csv"
    qc_df.to_csv(out_csv, index=False)

    print(f"[OK] S1 GRD finalization QC written to: {out_csv}")
    print()
    print("[SUMMARY]")
    print(qc_df["status"].value_counts(dropna=False).to_string())
    print()
    print("[BAND COUNTS]")
    print(qc_df["band_count"].value_counts(dropna=False).to_string())
    print()
    print("[GEOREF MODES]")
    if "vv_georef_mode" in qc_df.columns:
        print(qc_df["vv_georef_mode"].value_counts(dropna=False).to_string())
    print()
    print("[FAILED]")
    failed = qc_df[qc_df["status"] == "FAILED"]
    if len(failed) == 0:
        print("None")
    else:
        print(failed[["city", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
