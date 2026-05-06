#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/default.yaml}

echo "[STEP 0] Build input inventory"
python src/favela_postprocessing/00_build_inventory.py --config "$CONFIG"

echo "[STEP 1] Finalize Sentinel-2 products"
python src/favela_postprocessing/01_finalize_s2.py --config "$CONFIG"

echo "[STEP 2] Finalize Sentinel-1 products aligned to S2"
python src/favela_postprocessing/02_finalize_s1.py --config "$CONFIG"

echo "[STEP 3] Finalize labels aligned to S2"
python src/favela_postprocessing/03_finalize_labels.py --config "$CONFIG"

echo "[DONE] First three post-processing tasks completed."
echo "[QC] Check:"
echo "/media/HALLOPEAU/T7/post_processing_dataset/qc/"
