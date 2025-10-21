import pandas as pd
import numpy as np
import os

# =============================================================================
# Cancer Cell Motion Feature Extraction
# =============================================================================
# This script computes motion descriptors for tracked cancer cells.
# It uses the output of trackpy (cancer1_tracks.csv) and calculates
# features like mean speed, total displacement, mean squared displacement,
# directional persistence, and mean turning angle.
# These features are biologically meaningful markers of cancer cell behavior.

# -----------------------------------------------------------------------------
# Input/Output Paths
# -----------------------------------------------------------------------------
input_csv = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\tracking\cancer1_tracks.csv"
output_csv = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\features\cancer1_features.csv"

# Load tracking data
df = pd.read_csv(input_csv)

# -----------------------------------------------------------------------------
# Parameter Setup
# -----------------------------------------------------------------------------
frame_rate = 10.0  # fps — based on metadata
# Chosen because during metadata extraction, frame_rate was confirmed as 10 fps
# (10 frames per second = 0.1 s per frame)
# This allows us to convert per-frame displacement into physical (px/s) speed.

# -----------------------------------------------------------------------------
# Compute Per-Trajectory Motion Features
# -----------------------------------------------------------------------------
features = []

for pid, traj in df.groupby("particle"):
    traj = traj.sort_values("frame")
    if len(traj) < 2:
        continue  # Skip single-frame trajectories (no motion info)

    # Compute frame-to-frame displacement
    dx = np.diff(traj["x"])
    dy = np.diff(traj["y"])
    dist = np.sqrt(dx ** 2 + dy ** 2)

    # Mean speed (px/s)
    mean_speed = np.mean(dist * frame_rate)

    # Total displacement (px)
    total_disp = np.sqrt((traj["x"].iloc[-1] - traj["x"].iloc[0]) ** 2 +
                         (traj["y"].iloc[-1] - traj["y"].iloc[0]) ** 2)

    # Mean squared displacement (px^2)
    msd = np.mean(dist ** 2)

    # Directional persistence
    # ratio of net displacement to total path length
    path_length = np.sum(dist)
    persistence = total_disp / path_length if path_length > 0 else 0

    # Mean turning angle (degrees)
    # Measures how sharply the cell changes direction between steps.
    angles = []
    for i in range(1, len(dx)):
        v1 = np.array([dx[i - 1], dy[i - 1]])
        v2 = np.array([dx[i], dy[i]])
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1)  # numerical stability
            angle = np.degrees(np.arccos(cos_theta))
            angles.append(angle)
    mean_turn_angle = np.mean(angles) if len(angles) > 0 else np.nan

    features.append({
        "particle": pid,
        "mean_speed(px/s)": mean_speed,
        "total_displacement(px)": total_disp,
        "mean_squared_displacement(px^2)": msd,
        "directional_persistence": persistence,
        "mean_turn_angle(deg)": mean_turn_angle
    })

# Convert to DataFrame
features_df = pd.DataFrame(features)

# -----------------------------------------------------------------------------
# Summary Statistics (mean ± std)
# -----------------------------------------------------------------------------
summary = features_df.describe().loc[["mean", "std"]].T
summary["mean ± std"] = summary["mean"].round(2).astype(str) + " ± " + summary["std"].round(2).astype(str)

print("\n[SUMMARY STATISTICS]\n")
print(summary[["mean ± std"]])

# -----------------------------------------------------------------------------
# Save Results
# -----------------------------------------------------------------------------
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
features_df.to_csv(output_csv, index=False)
print(f"\n[INFO] Computed features for {len(features_df)} cells.")
print(f"[INFO] Results saved to '{output_csv}'.")
