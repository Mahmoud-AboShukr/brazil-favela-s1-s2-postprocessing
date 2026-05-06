from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def count_eq(df, col, value):
    if col not in df.columns:
        return 0
    return int((df[col] == value).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])
    qc_dir = output_root / "qc"
    metadata_dir = output_root / "metadata"

    inventory = read_csv(metadata_dir / "city_input_inventory.csv")
    s2 = read_csv(qc_dir / "s2_finalization_qc.csv")
    s1 = read_csv(qc_dir / "s1_grd_finalization_qc.csv")
    labels = read_csv(qc_dir / "label_finalization_qc.csv")
    align = read_csv(qc_dir / "alignment_qc.csv")

    summary = inventory[["city", "source_group", "has_s1"]].copy()

    s2_small = s2[["city", "s2_status", "band_count", "s2_output"]].copy()
    s2_small = s2_small.rename(columns={"band_count": "s2_band_count"})
    summary = summary.merge(s2_small, on="city", how="left")

    s1_small = s1[["city", "status", "band_count", "s1_output", "error"]].copy()
    s1_small = s1_small.rename(
        columns={
            "status": "s1_status",
            "band_count": "s1_band_count",
            "error": "s1_error",
        }
    )
    summary = summary.merge(s1_small, on="city", how="left")

    label_small = labels[
        [
            "city",
            "label_status",
            "original_status",
            "label_positive_pixels",
            "label_positive_percent",
            "original_positive_pixels",
            "original_positive_percent",
        ]
    ].copy()
    summary = summary.merge(label_small, on="city", how="left")

    align_small = align[
        [
            "city",
            "city_status",
            "s2_exists",
            "s1_exists",
            "label_exists",
            "s1_alignment_ok",
            "label_alignment_ok",
            "original_alignment_ok",
        ]
    ].copy()
    summary = summary.merge(align_small, on="city", how="left")

    def final_status(row):
        if row["city_status"] == "OK_S2_S1_LABEL":
            return "complete_s1_s2_label"
        if row["city_status"] == "OK_S2_LABEL_S1_MISSING":
            return "s2_label_only_s1_missing"
        return "check_required"

    summary["final_city_status"] = summary.apply(final_status, axis=1)

    out_csv = qc_dir / "final_postprocessing_summary.csv"
    summary.to_csv(out_csv, index=False)

    total = len(summary)
    s2_success = count_eq(s2, "s2_status", "OK") + count_eq(s2, "s2_status", "SKIPPED_EXISTS")
    s1_success = count_eq(s1, "status", "OK") + count_eq(s1, "status", "SKIPPED_EXISTS")
    label_success = count_eq(labels, "label_status", "OK") + count_eq(labels, "label_status", "SKIPPED_EXISTS")
    original_success = count_eq(labels, "original_status", "OK") + count_eq(labels, "original_status", "SKIPPED_EXISTS")
    s2_label_ok = count_eq(align, "label_alignment_ok", True)
    s2_s1_ok = count_eq(align, "s1_alignment_ok", True)
    complete = count_eq(summary, "final_city_status", "complete_s1_s2_label")
    s2_only = count_eq(summary, "final_city_status", "s2_label_only_s1_missing")

    missing_s1 = summary.loc[
        summary["final_city_status"] == "s2_label_only_s1_missing", "city"
    ].tolist()

    out_md = qc_dir / "final_postprocessing_summary.md"

    lines = []
    lines.append("# Final Post-processing QC Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"Output root: {output_root}")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Total cities: {total}")
    lines.append(f"- Finalized Sentinel-2 products: {s2_success} / {total}")
    lines.append(f"- Finalized Sentinel-1 GRD products: {s1_success} / {total}")
    lines.append(f"- Finalized label masks: {label_success} / {total}")
    lines.append(f"- Auxiliary original polygon masks: {original_success} / {total}")
    lines.append(f"- S2 + label alignment OK: {s2_label_ok} / {total}")
    lines.append(f"- S2 + S1 alignment OK: {s2_s1_ok} / {total}")
    lines.append(f"- Complete S1 + S2 + label cities: {complete} / {total}")
    lines.append(f"- S2 + label only cities: {s2_only} / {total}")
    lines.append("")
    lines.append("## Known issue")
    lines.append("")
    if missing_s1:
        lines.append("The following city is missing Sentinel-1 GRD:")
        for city in missing_s1:
            lines.append(f"- {city}")
    else:
        lines.append("No city is missing Sentinel-1 GRD.")
    lines.append("")
    lines.append("This is a data availability issue, not a processing or alignment failure.")
    lines.append("")
    lines.append("## Per-city status")
    lines.append("")
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['city']}: {row['final_city_status']} "
            f"(S2={row['s2_status']}, S1={row['s1_status']}, label={row['label_status']})"
        )

    out_md.write_text("\n".join(lines))

    print(f"[OK] Wrote CSV: {out_csv}")
    print(f"[OK] Wrote Markdown: {out_md}")
    print()
    print("[FINAL CITY STATUS]")
    print(summary["final_city_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
