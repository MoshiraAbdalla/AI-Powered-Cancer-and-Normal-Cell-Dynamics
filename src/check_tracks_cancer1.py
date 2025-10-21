import pandas as pd

# Load tracked data
df = pd.read_csv(r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\tracking\cancer1_tracks.csv")

print("[INFO] Tracking Summary")
print(f"Total unique cells tracked: {df['particle'].nunique()}")
print(f"Average trajectory length: {df.groupby('particle')['frame'].count().mean():.2f} frames")
print(f"Max trajectory length: {df.groupby('particle')['frame'].count().max()} frames")
print(f"Min trajectory length: {df.groupby('particle')['frame'].count().min()} frames")

print("\nTop 10 longest trajectories:")
print(df.groupby('particle')['frame'].count().sort_values(ascending=False).head(10))
