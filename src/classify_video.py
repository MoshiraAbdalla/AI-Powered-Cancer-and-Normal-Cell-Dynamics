import pandas as pd

# Path to extracted features (change per video)
features_path = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\features\normal_features.csv"
features_path1 = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\features\cancer1_features.csv"

# Load data
df = pd.read_csv(features_path1)

# Compute average values
mean_speed = df['mean_speed(px/s)'].mean()
total_disp = df['total_displacement(px)'].mean()
mean_angle = df['mean_turn_angle(deg)'].mean()

print("[INFO] Computed Statistics:")
print(f"Mean Speed: {mean_speed:.2f} px/s")
print(f"Total Displacement: {total_disp:.2f} px")
print(f"Mean Turn Angle: {mean_angle:.2f}°")

# --- Thresholds (derived from dataset statistics) ---
speed_threshold = 49
angle_threshold = 80
disp_threshold = 21

# --- Classification Logic ---
if (mean_speed > speed_threshold) and (mean_angle > angle_threshold) and (total_disp < disp_threshold):
    prediction = "Cancer Cell"
else:
    prediction = "Normal Cell"

print("\n[RESULT] Classification:", prediction)
