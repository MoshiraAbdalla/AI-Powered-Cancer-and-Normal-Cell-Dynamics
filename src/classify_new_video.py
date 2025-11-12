import cv2
import os
import numpy as np
import pandas as pd
from skimage import io, filters, morphology, measure
import trackpy as tp
from trackpy.linking.utils import SubnetOversizeException

# ---------------- CONFIGURATION ----------------
video_path = r"Dataset\Normal2.avi"
# video_path = r"Dataset\Cancer2.mp4"
frames_dir = r"Results\frames_temp_N2"
seg_dir = r"Results\segmentation_temp_N2"
# frames_dir = r"Results\frames_temp_c2"
# seg_dir = r"Results\segmentation_temp_c2"
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)

# ---------------- 1. FRAME EXTRACTION ----------------
print("\n[INFO] Extracting frames...")
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
res = (int(cap.get(3)), int(cap.get(4)))
print(f"[INFO] Video Metadata:\nFrames: {frame_count}, FPS: {fps}, Resolution: {res}")

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    io.imsave(os.path.join(frames_dir, f"frame_{i:03d}.png"), gray)
    i += 1
cap.release()
print(f"[INFO] Saved {i} frames to '{frames_dir}'")

# ---------------- 2. SEGMENTATION ----------------
print("\n[INFO] Performing segmentation...")
for f in os.listdir(frames_dir):
    img = io.imread(os.path.join(frames_dir, f))
    thr = filters.threshold_otsu(img)
    mask = img > thr
    mask = morphology.remove_small_objects(mask, 40)
    mask = morphology.remove_small_holes(mask, 40)
    io.imsave(os.path.join(seg_dir, f), (mask * 255).astype(np.uint8))
print(f"[INFO] Segmentation complete – saved masks to '{seg_dir}'")

# ---------------- 3. TRACKING ----------------
print("\n[INFO] Tracking cells...")
all_data = []
for idx, f in enumerate(sorted(os.listdir(seg_dir))):
    img = io.imread(os.path.join(seg_dir, f))
    labeled = measure.label(img > 0)
    props = measure.regionprops(labeled)
    for p in props:
        y, x = p.centroid
        all_data.append([x, y, idx])
df = pd.DataFrame(all_data, columns=["x", "y", "frame"])

# Handle dense tracking with adaptive retry
search_range = 25
memory = 5

try:
    linked = tp.link_df(df, search_range=search_range, memory=memory)
except SubnetOversizeException as e:
    print(f"[WARN] {e}. Retrying with smaller search range...")
    linked = tp.link_df(df, search_range=15, memory=memory)

print(f"[INFO] Tracking complete – {linked['particle'].nunique()} cells identified.")

# ---------------- 4. FEATURE EXTRACTION ----------------
print("\n[INFO] Computing motion features...")
features = []
for pid, group in linked.groupby("particle"):
    if len(group) < 3:
        continue
    group = group.sort_values("frame")
    dx = np.diff(group["x"])
    dy = np.diff(group["y"])
    speed = np.mean(np.sqrt(dx**2 + dy**2))
    disp = np.sqrt((group["x"].iloc[-1] - group["x"].iloc[0])**2 +
                   (group["y"].iloc[-1] - group["y"].iloc[0])**2)
    ang = np.mean(np.abs(np.degrees(np.arctan2(dy, dx))))
    features.append([pid, speed, disp, ang])
feat_df = pd.DataFrame(features, columns=["particle", "speed", "disp", "angle"])
print(f"[INFO] Computed features for {len(feat_df)} cells.")

mean_speed = feat_df["speed"].mean()
mean_disp = feat_df["disp"].mean()
mean_angle = feat_df["angle"].mean()

print(f"\n[INFO] Video-Level Statistics:"
      f"\nMean Speed: {mean_speed:.2f} px/s"
      f"\nTotal Displacement: {mean_disp:.2f} px"
      f"\nMean Turn Angle: {mean_angle:.2f}°")

# ---------------- 5. CLASSIFICATION (VOTING MODEL) ----------------
# Reference statistics
norm = {'speed': 35.96, 'disp': 26.08, 'angle': 69.7}
canc = {'speed': 61.95, 'disp': 15.48, 'angle': 90.89}

# Mid thresholds between classes
th_speed = (norm['speed'] + canc['speed']) / 2
th_disp = (norm['disp'] + canc['disp']) / 2
th_angle = (norm['angle'] + canc['angle']) / 2

votes = {"Cancer": 0, "Normal": 0}

# Speed: higher = cancer
if mean_speed > th_speed:
    votes["Cancer"] += 1
else:
    votes["Normal"] += 1

# Displacement: higher = normal
if mean_disp > th_disp:
    votes["Normal"] += 1
else:
    votes["Cancer"] += 1

# Turn angle: higher = cancer
if mean_angle > th_angle:
    votes["Cancer"] += 1
else:
    votes["Normal"] += 1

# Final decision
classification = "Cancer Cell" if votes["Cancer"] > votes["Normal"] else "Normal Cell"
cancer_likeness = votes["Cancer"] / 3

print(f"\n[RESULT] Classification: {classification}")
print(f"[INFO] Cancer-Likeness Score: {cancer_likeness:.2f}")
print(f"[INFO] Feature Votes: {votes}")
