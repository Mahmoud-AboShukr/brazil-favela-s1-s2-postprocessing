# Brazil Favela S1/S2 Post-processing

Post-processing pipeline for preparing Sentinel-1, Sentinel-2, and favela label masks for a deep-learning-ready Brazil favela segmentation dataset.

The pipeline keeps raw/intermediate inputs untouched and writes all processed outputs to a separate directory.

## Main inputs

Default input root:

```text
/media/HALLOPEAU/T7/my_processed_data



Expected inputs include:

Sentinel-2 cloud-reduced city composites
Sentinel-1 RTC/GRD city products
refined favela polygons
clean original favela polygons
Main output

Default output root:

/media/HALLOPEAU/T7/post_processing_dataset
First processing tasks
Finalize Sentinel-2 products while keeping all downloaded bands.
Finalize Sentinel-1 products and align them to the Sentinel-2 grid.
Rasterize final favela labels on the same Sentinel-2 grid.
Design principle

The main dataset should stay simple:

S2 + S1 + final label

Auxiliary masks, cloud masks, original polygon masks, and QC files are stored separately for reproducibility.
