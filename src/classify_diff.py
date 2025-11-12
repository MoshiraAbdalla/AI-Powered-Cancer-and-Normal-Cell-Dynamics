import cv2
import os
import numpy as np
import pandas as pd
from skimage import io, filters, morphology, measure
import trackpy as tp
from trackpy.linking.utils import SubnetOversizeException
import numpy as np, matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Matplotlib offscreen for saving the DEA figure ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- CONFIGURATION ----------------
# video_path = r"Dataset\Normal3.avi"

# video_path = r"Dataset\Normal2.avi"
# video_path = r"Dataset\Cancer2.mp4"
# video_path = r"Dataset\Cancer3.avi"
# frames_dir = r"Results\frames_temp_c3"
# seg_dir    = r"Results\segmentation_temp_c3"

video_path = r"Dataset\Normal.mp4"

# video_path = r"Dataset\Cancer1.mp4" # Best diffusion behavior
# frames_dir = r"Results\frames_temp_N2"
# seg_dir    = r"Results\segmentation_temp_N2"
# frames_dir = r"Results\frames_temp_N3"
# seg_dir    = r"Results\segmentation_temp_N3"
# frames_dir = r"Results\frames_temp_c2"
# seg_dir    = r"Results\segmentation_temp_c2"
frames_dir = r"Results\frames_temp_N1"
seg_dir    = r"Results\segmentation_temp_N1"
# frames_dir = r"Results\frames_temp_c1"
# seg_dir    = r"Results\segmentation_temp_c1"

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)

# Parent results folder (to drop summary images)
out_dir = os.path.dirname(frames_dir) if os.path.dirname(frames_dir) else "Results"
os.makedirs(out_dir, exist_ok=True)



def _theil_sen(x, y):
    """Robust slope/intercept for short noisy series."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 3:
        # fall back to least-squares
        A = np.vstack([x, np.ones(n)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(b)
    # all pairwise slopes
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    mask = np.triu(np.ones((n, n), bool), 1) & (np.abs(dx) > 0)
    slopes = (dy[mask] / dx[mask])
    m = np.median(slopes)
    b = np.median(y - m * x)
    return float(m), float(b)



def dea_from_tracks_paper_exact(linked_df, frame_col="frame",
                                xcols=("x","y"), min_traj_len=40,
                                t_points=90, nbins=70, 
                                min_counts_per_bin=0, out_png=None):
    """
    DEA with correct Δt:
      • resample each track to a regular per-frame grid (linear interp)
      • use fixed histogram bins across all t
      • fit only adequately sampled mid/upper t using robust Theil–Sen
    Returns: (t_grid, S_t, delta)
    """

    if linked_df is None or linked_df.empty or "particle" not in linked_df.columns:
        return None

    # --- resample each track to every integer frame ---
    series = []
    for _, g in linked_df.groupby("particle"):
        g = g.sort_values(frame_col)
        if len(g) < min_traj_len:
            continue
        f = g[frame_col].to_numpy(int)
        full = np.arange(f[0], f[-1] + 1)
        xi = np.interp(full, f, g[xcols[0]].to_numpy(float))
        yi = np.interp(full, f, g[xcols[1]].to_numpy(float))
        s = np.c_[xi, yi]
        if len(s) >= min_traj_len:
            series.append(s)

    if not series:
        return None

    # --- t grid (log-spaced, deduped after rounding) ---
    med_len = int(np.median([len(s) for s in series]))
    t_max = max(6, min(med_len // 2, 64))
    t_grid = np.unique(np.clip(np.round(np.geomspace(2, t_max, t_points)).astype(int), 2, None))

    # --- pooled deltas to set FIXED bin edges (robust central range) ---
    pooled = []
    for s in series:
        for t in t_grid:
            if len(s) > t:
                d = s[t:] - s[:-t]
                pooled.append(d[:, 0]); pooled.append(d[:, 1])
    pooled = np.concatenate(pooled)
    q1, q99 = np.percentile(pooled, [1, 99])
    span = max(q99 - q1, 1e-6)
    lo = q1 - 0.1 * span
    hi = q99 + 0.1 * span
    bin_edges = np.linspace(lo, hi, nbins + 1)

    # --- entropy S(t) with fixed bins ---
    S_vals, Ns = [], []
    for t in t_grid:
        deltas = []
        for s in series:
            if len(s) > t:
                d = s[t:] - s[:-t]
                deltas.append(d[:, 0]); deltas.append(d[:, 1])
        if not deltas:
            S_vals.append(np.nan); Ns.append(0); continue
        deltas = np.concatenate(deltas)
        counts, _ = np.histogram(deltas, bins=bin_edges)
        tot = int(counts.sum()); Ns.append(tot)
        if tot == 0:
            S_vals.append(np.nan); continue
        p = counts[counts > 0].astype(float) / tot
        S_vals.append(float(-(p * np.log(p)).sum()))
    S_vals = np.array(S_vals, float)
    Ns = np.array(Ns, int)

    # --- choose reliable fit range (avoid sparse tail) ---
    valid = np.isfinite(S_vals) & (Ns / nbins >= min_counts_per_bin)
    if valid.sum() < 6:
        delta = np.nan
        A = np.nan
        start = 0
    else:
        idx = np.where(valid)[0]
        start = idx[len(idx) // 2]          # upper half of valid points
        x = np.log(t_grid[start:])
        y = S_vals[start:]
        delta, A = _theil_sen(x, y)         # robust slope/intercept

    # --- plot ---
    if out_png is not None:
        fig = plt.figure(figsize=(7.5, 6), dpi=160)
        ax = fig.add_subplot(111)
        ax.scatter(t_grid, S_vals, s=30, alpha=0.9)
        if np.isfinite(delta):
            # ax.plot(t_grid[start:], A + delta * np.log(t_grid[start:]), lw=2)
            ttl = f"Diffusion Entropy Analysis (δ ≈ {delta:.3f})"
        else:
            ttl = "Diffusion Entropy Analysis"
        # ax.set_xscale("log")
        # # ax.xaxis.set_minor_formatter(matplotlib.ticker.LogFormatter())   # minor tick labels
        # ax.xaxis.set_minor_formatter(matplotlib.ticker.LogFormatter())   # minor tick labels
        # ax.minorticks_on()
        # ax.tick_params(axis='x', which='minor', length=4)
        # ax.set_xlabel("Time window t")

        ax.set_xscale("log")

        # Force readable ticks across one decade
        ax.set_xticks([2, 3, 5, 10, 20, 30])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())   # show as numbers
        ax.tick_params(axis='x', which='major', length=6)

        ax.set_xlabel("Time window t")

        ax.set_ylabel("S(t)")
        ax.set_title(ttl)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    return t_grid, S_vals, float(delta) if np.isfinite(delta) else np.nan


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
for f in sorted(os.listdir(frames_dir)):
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
    for p in measure.regionprops(labeled):
        y, x = p.centroid
        all_data.append([x, y, idx])
df = pd.DataFrame(all_data, columns=["x", "y", "frame"])

search_range = 25
memory = 5
try:
    linked = tp.link_df(df, search_range=search_range, memory=memory)
except SubnetOversizeException as e:
    print(f"[WARN] {e}. Retrying with smaller search range...")
    linked = tp.link_df(df, search_range=15, memory=memory)

print(f"[INFO] Tracking complete – {linked['particle'].nunique()} cells identified.")

# ---------------- 4. FEATURE EXTRACTION (+ Persistence) ----------------
print("\n[INFO] Computing motion features...")
features = []
for pid, group in linked.groupby("particle"):
    if len(group) < 3:
        continue
    group = group.sort_values("frame")
    x = group["x"].to_numpy()
    y = group["y"].to_numpy()
    dx = np.diff(x)
    dy = np.diff(y)
    step = np.sqrt(dx**2 + dy**2)

    speed = float(np.mean(step))*fps  # px/s
    disp  = float(np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))
    ang   = float(np.mean(np.abs(np.degrees(np.arctan2(dy, dx)))))

    path_len = float(np.sum(step))
    persistP = float(disp / path_len) if path_len > 0 else 0.0  # D/L

    features.append([pid, speed, disp, ang, path_len, persistP])

feat_df = pd.DataFrame(
    features, columns=["particle", "speed", "disp", "angle", "path_len", "persist_P"]
)
print(f"[INFO] Computed features for {len(feat_df)} cells.")

mean_speed = float(feat_df["speed"].mean()) if len(feat_df) else 0.0
mean_disp  = float(feat_df["disp"].mean())  if len(feat_df) else 0.0
mean_angle = float(feat_df["angle"].mean()) if len(feat_df) else 0.0
mean_P     = float(feat_df["persist_P"].mean()) if len(feat_df) else 0.0

print(
    f"\n[INFO] Video-Level Statistics:"
    f"\nMean Speed: {mean_speed:.2f} px/s"
    f"\nTotal Displacement: {mean_disp:.2f} px"
    f"\nMean Turn Angle: {mean_angle:.2f}°"
    f"\nMean Persistence (D/L): {mean_P:.3f}"
)



# ---------------- 5. CLASSIFICATION (VOTING MODEL + DEA vote) ----------------
# ---------------- 5. CLASSIFICATION (VOTING MODEL + DEA + Persistence) ----------------
# Reference statistics
 # 5) Voting thresholds
norm = {'speed': 35.96, 'disp': 26.08, 'angle': 69.7}
canc = {'speed': 61.95, 'disp': 15.48, 'angle': 90.89}
# th_speed = (norm['speed'] + canc['speed']) / 2
th_disp  = (norm['disp']  + canc['disp'])  / 2
# th_angle = (norm['angle'] + canc['angle']) / 2
th_speed=46.5397
# th_disp=29.2267
th_angle=74.8213
th_persist = 0.50
# th_dea = 0.50

votes = {"Cancer": 0, "Normal": 0}
total_votes = 0

# Speed: higher = cancer
votes["Cancer" if mean_speed > th_speed else "Normal"] += 1; total_votes += 1

# Displacement: higher = normal
votes["Normal" if mean_disp > th_disp else "Cancer"] += 1; total_votes += 1

# Turn angle: higher = cancer
votes["Cancer" if mean_angle > th_angle else "Normal"] += 1; total_votes += 1

# ---------------- 4b. DIFFUSION ENTROPY ANALYSIS ----------------

#_______________________
dea_png = os.path.join(out_dir, "diffusion_entropy_dense.png")
dea_csv = os.path.join(out_dir, "diffusion_entropy_dense.csv")

res = dea_from_tracks_paper_exact(linked, out_png=dea_png)

# delta = np.nan
# if res is not None:
#     _, _, delta = res
#     if np.isfinite(delta):
#         print(f"[INFO] DEA slope δ = {delta:.3f}")
#         mu_S = 1.0 + 1.0/delta
#         print(f"[INFO] μ_S (from DEA) = {mu_S:.3f}")
#         # DEA vote (your rule):
#         votes["Cancer" if delta > 0.5 else "Normal"] += 1
#     else:
#         print("[WARN] DEA tail fit not reliable; vote skipped.")
# else:
#     print("[WARN] DEA skipped (insufficient tracks).")

# print(f"[INFO] Saved dense DEA plot: {dea_png}")
# print(f"[INFO] Saved dense DEA table: {dea_csv}")

if res is not None:
    _, _, delta = res
    if np.isfinite(delta):
        print(f"[INFO] DEA slope δ = {delta:.3f}")
        mu_S = 1.0 + 1.0/delta
        print(f"[INFO] μ_S (from DEA) = {mu_S:.3f}")
        votes["Cancer" if delta > 0.5 else "Normal"] += 1
        total_votes += 1               # <-- missing in your code
    else:
        print("[WARN] DEA vote skipped (no reliable asymptotic slope).")


# # DEA vote: δ > 0.5 ⇒ Cancer (only if we have a finite slope)
# if np.isfinite(delta):
#     votes["Cancer" if delta > 0.5 else "Normal"] += 1
#     total_votes += 1
# else:
#     print("[WARN] DEA vote skipped (δ is NaN).")

# Persistence vote: P > 0.5 ⇒ Normal, otherwise Cancer
votes["Normal" if mean_P > th_persist else "Cancer"] += 1
total_votes += 1

# Final decision
classification = "Cancer Cell" if votes["Cancer"] > votes["Normal"] else "Normal Cell"
cancer_likeness = votes["Cancer"] / max(total_votes, 1)

print(f"\n[RESULT] Classification: {classification}")
print(f"[INFO] Cancer-Likeness Score: {cancer_likeness:.2f}")
print(f"[INFO] Feature Votes: {votes}")
print(f"[INFO] Thresholds -> speed:{th_speed:.2f}, disp:{th_disp:.2f}, angle:{th_angle:.2f}, "
      f"persistence:{th_persist:.3f}, DEA:0.50")
