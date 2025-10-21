import os
import pandas as pd
import numpy as np
import trackpy as tp
from skimage import io, measure

def track_cells(segmentation_dir, output_csv):
    """
    Detect cell centroids from binary segmentation masks and track them across frames.
    Saves trajectories (x, y, frame, particle ID) to a CSV file.
    """

    # --- Step 1: Gather segmentation masks ---
    frame_files = sorted([f for f in os.listdir(segmentation_dir) if f.endswith(".png")])
    all_cells = []

    for i, filename in enumerate(frame_files):
        img_path = os.path.join(segmentation_dir, filename)
        mask = io.imread(img_path)
        labeled = measure.label(mask)  # Label connected regions (each cell = unique label)
        props = measure.regionprops(labeled)

        for p in props:
            y, x = p.centroid
            all_cells.append({'frame': i, 'x': x, 'y': y})

    # --- Step 2: Convert detections into a DataFrame ---
    df = pd.DataFrame(all_cells)
    print(f"[INFO] Detected {len(df)} cell centroids across {len(frame_files)} frames.")

    # --- Step 3: Track cells with TrackPy ---
    # search_range = max distance (in pixels) a cell can move between consecutive frames
    # memory = how many frames a cell can disappear and still be reconnected later
    tracked = tp.link_df(df, search_range=30, memory=2)

    print(f"[INFO] Tracking complete — {tracked['particle'].nunique()} unique cells identified.")

    # --- Step 4: Save results ---
    tracked.to_csv(output_csv, index=False)
    print(f"[INFO] Saved tracking data to '{output_csv}'.")

    return tracked


if __name__ == "__main__":
    segmentation_dir = r"./Results/segmentation"
    output_csv = r"./Results/tracking/normal_tracks.csv"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    tracks = track_cells(segmentation_dir, output_csv)
