import pandas as pd
import numpy as np
import os

def compute_motion_features(tracks_csv, output_csv):
    """
    Compute quantitative motion features for each tracked cell trajectory.
    Features include:
    - Mean speed (µm/s or pixels/frame)
    - Total displacement (net distance from start to end)
    - Mean squared displacement (MSD)
    - Directional persistence (straightness ratio)
    - Average turning angle (path tortuosity)
    """

    # --- Step 1: Load trajectory data ---
    df = pd.read_csv(tracks_csv)

    # Check necessary columns exist
    if not {'frame', 'x', 'y', 'particle'}.issubset(df.columns):
        raise ValueError("CSV must contain columns: frame, x, y, particle")

    # --- Step 2: Compute per-frame displacement for each cell ---
    df['dx'] = df.groupby('particle')['x'].diff()
    df['dy'] = df.groupby('particle')['y'].diff()
    df['step_displacement'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # Justification:
    # The Euclidean distance between consecutive centroids (Δx, Δy)
    # represents how far the cell moved between frames.
    # This is the base metric for speed and directionality.

    # --- Step 3: Frame rate (from metadata) ---
    # Using your measured FPS = 6.98 (from metadata step)
    fps = 6.98
    time_per_frame = 1 / fps  # ≈ 0.143 s/frame

    # --- Step 4: Compute speed ---
    # Convert displacement (in pixels) to speed (pixels per second)
    df['speed'] = df['step_displacement'] / time_per_frame

    # --- Step 5: Compute per-cell aggregated features ---
    features = []
    for pid, group in df.groupby('particle'):
        group = group.dropna()

        if len(group) < 3:
            # Ignore very short trajectories (1–2 frames)
            continue

        # Total displacement (straight-line distance start → end)
        total_disp = np.sqrt((group['x'].iloc[-1] - group['x'].iloc[0])**2 +
                             (group['y'].iloc[-1] - group['y'].iloc[0])**2)

        # Path length (sum of all frame-to-frame movements)
        path_length = group['step_displacement'].sum()

        # Directional persistence (straightness ratio)
        # ratio close to 1 = linear motion; lower = random wandering
        persistence = total_disp / path_length if path_length > 0 else np.nan

        # Mean speed (pixels/sec)
        mean_speed = group['speed'].mean()

        # Mean Squared Displacement (MSD)
        msd = np.mean(group['step_displacement'] ** 2)

        # Turning angle (mean change in direction)
        dx = group['dx'].to_numpy()
        dy = group['dy'].to_numpy()
        angles = np.arctan2(dy, dx)
        dtheta = np.diff(angles)
        # Normalize angles to [-π, π] range
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        mean_turn_angle = np.degrees(np.mean(np.abs(dtheta)))

        features.append({
            'particle': pid,
            'frames_tracked': len(group),
            'mean_speed(px/s)': round(mean_speed, 3),
            'total_displacement(px)': round(total_disp, 3),
            'mean_squared_displacement(px²)': round(msd, 3),
            'directional_persistence': round(persistence, 3),
            'mean_turn_angle(deg)': round(mean_turn_angle, 2)
        })

    features_df = pd.DataFrame(features)
    features_df.to_csv(output_csv, index=False)

    print(f"[INFO] Computed features for {len(features_df)} cells.")
    print(f"[INFO] Results saved to '{output_csv}'.")

    return features_df


if __name__ == "__main__":
    # Absolute paths
    tracks_csv = r"./Results/tracking/normal_tracks.csv"
    output_csv = r"./Results/features/normal_features.csv"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    compute_motion_features(tracks_csv, output_csv)
