from pathlib import Path
import re

import pandas as pd
import tifffile as tiff
from skimage.measure import regionprops_table
# This file is in:
# spatial-ihc-feature-lab/src/data/extract_mibi_centroids.py
# parents[0] = src/data
# parents[1] = src
# parents[2] = spatial-ihc-feature-lab
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASK_DIR = PROJECT_ROOT / "data" / "raw" / "mibi_tnbc" / "labeled_cell_masks"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

mask_files = sorted(list(MASK_DIR.glob("*.tif")) + list(MASK_DIR.glob("*.tiff")))

all_centroids = []

mask_files = sorted(list(MASK_DIR.glob("*.tif")) + list(MASK_DIR.glob("*.tiff")))

print("Project root:", PROJECT_ROOT)
print("Mask directory:", MASK_DIR)
print("Mask directory exists:", MASK_DIR.exists())
print("Number of mask files found:", len(mask_files))

if len(mask_files) == 0:
    raise FileNotFoundError(
        f"No .tif or .tiff files found in: {MASK_DIR}\n"
        "Check that the folder name is correct and that the TIFF masks are inside it."
    )

# -----------------------------
# Extract centroids
# -----------------------------
for mask_path in mask_files:
    # Extract sample number from names like p1_labeledcellData.tiff
    match = re.search(r"p(\d+)", mask_path.name)

    if match is None:
        print(f"Skipping file because sample_id could not be parsed: {mask_path.name}")
        continue

    sample_id = int(match.group(1))

    print(f"Processing sample {sample_id}: {mask_path.name}")

    mask = tiff.imread(mask_path)

    # regionprops_table calculates properties for every labeled object.
    # label 0 is background and is automatically ignored.
    props = regionprops_table(
        mask,
        properties=["label", "area", "centroid"]
    )

    sample_centroids = pd.DataFrame(props)

    sample_centroids = sample_centroids.rename(columns={
        "label": "cell_label",
        "area": "area_pixels",
        "centroid-0": "centroid_y",
        "centroid-1": "centroid_x",
    })

    sample_centroids["sample_id"] = sample_id

    sample_centroids = sample_centroids[
        ["sample_id", "cell_label", "centroid_x", "centroid_y", "area_pixels"]
    ]

    all_centroids.append(sample_centroids)

centroids_df = pd.concat(all_centroids, ignore_index=True)

output_path = PROCESSED_DIR / "mibi_cell_centroids.csv"
centroids_df.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Centroid table shape:", centroids_df.shape)

centroids_df.head()