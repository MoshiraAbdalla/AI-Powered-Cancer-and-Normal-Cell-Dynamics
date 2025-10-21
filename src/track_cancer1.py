import cv2
import os
import pandas as pd
import trackpy as tp
import numpy as np
from skimage.measure import label, regionprops

# --- Input directories ---
segmentation_dir = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\segmentation_cancer1"
output_csv = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\tracking\cancer1_tracks.csv"

# --- Parameters ---
search_range = 20   # reduced for stability
memory = 3

# --- Allow larger subnetworks ---
tp.linking.Linker.MAX_SUB_NET_SIZE = 100

# --- Step 1: Read segmented masks and extract centroids ---
frames = sorted(os.listdir(segmentation_dir))
all_detections = []

for frame_num, filename in enumerate(frames):
    frame_path = os.path.join(segmentation_dir, filename)
    mask = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    labeled_mask = label(mask > 0)
    props = regionprops(labeled_mask)

    # Skip frames that are too dense (optional safeguard)
    if len(props) > 200:
        print(f"[WARNING] Skipping dense frame {frame_num} with {len(props)} detections")
        continue

    for prop in props:
        y, x = prop.centroid
        all_detections.append([x, y, frame_num])

df = pd.DataFrame(all_detections, columns=["x", "y", "frame"])

# --- Step 2: Link detections ---
linked = tp.link_df(df, search_range=search_range, memory=memory, adaptive_stop=10, adaptive_step=0.95)

# --- Step 3: Save results ---
linked.to_csv(output_csv, index=False)

num_cells = linked["particle"].nunique()
print(f"[INFO] Tracking complete — {num_cells} unique cancer cells identified.")
print(f"[INFO] Tracking data saved to: {output_csv}")
