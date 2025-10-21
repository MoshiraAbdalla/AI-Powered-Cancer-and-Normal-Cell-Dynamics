import pandas as pd

tracks = pd.read_csv(r"./Results/tracking/normal_tracks.csv")

# Count how many frames each cell was tracked
track_lengths = tracks.groupby('particle')['frame'].count()

print("[INFO] Tracking Summary")
print(f"Total unique cells tracked: {tracks['particle'].nunique()}")
print(f"Average trajectory length: {track_lengths.mean():.2f} frames")
print(f"Max trajectory length: {track_lengths.max()} frames")
print(f"Min trajectory length: {track_lengths.min()} frames")

# Optional: show top 10 longest tracks
print("\nTop 10 longest trajectories:")
print(track_lengths.sort_values(ascending=False).head(10))
