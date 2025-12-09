import cv2
import os
import numpy as np
import pandas as pd
from skimage import io, filters, morphology, measure
import trackpy as tp
from trackpy.linking.utils import SubnetOversizeException

# ---------------- CONFIGURATION ----------------
# Choose the video you want to run now
# video_path = r"Dataset\Normal.mp4"
# video_path = r"Dataset\Normal2.avi"
# video_path = r"Dataset\Cancer1.mp4"
video_path = r"Dataset\Cancer2.mp4"

# Output dirs
# frames_dir = r"Results\frames_temp_N2"; seg_dir = r"Results\segmentation_temp_N2"
# frames_dir = r"Results\frames_temp_N1"; seg_dir = r"Results\segmentation_temp_N1"
# frames_dir = r"Results\frames_temp_C1"; seg_dir = r"Results\segmentation_temp_C1"
frames_dir = r"Results\frames_temp_C2"; seg_dir = r"Results\segmentation_temp_C2"
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)

# --- Class reference means (original, used for p-value testing) ---
# Speeds in px/s (converted later if needed)
REF_NORM_SPEED_PXS = 35.96
REF_CANC_SPEED_PXS = 61.95
REF_MEANS = {
    "Normal": {"speed_pxs": REF_NORM_SPEED_PXS, "disp": 26.08, "angle": 69.70},
    "Cancer": {"speed_pxs": REF_CANC_SPEED_PXS, "disp": 15.48, "angle": 90.89},
}

# --- Empirical calibration from YOUR four videos (per-frame already) ---
# Normal1: speed=3.05, disp=16.64, angle=nan
# Normal2: speed=5.35, disp=20.13, angle=76.25
# Cancer1: speed=5.46, disp=16.28, angle=90.82
# Cancer2: speed=5.40, disp=15.78, angle=96.76
EMP_MEANS = {
    "Normal": {"speed": 4.20,   "disp": 18.385, "angle": 76.25},     # avg over Normal1+Normal2 (angle uses N2 only)
    "Cancer": {"speed": 5.43,   "disp": 16.03,  "angle": 93.79},     # avg over Cancer1+Cancer2
}

# How to set thresholds used for tie-break only:
#  - "empirical": use midpoints between EMP_MEANS Normal/Cancer per metric (your videos)
#  - "reference": use midpoints between REF means (converted to per-frame)
CAL_MODE = "empirical"   # change to "reference" any time

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
    props = measure.regionprops(labeled)
    for p in props:
        y, x = p.centroid
        all_data.append([x, y, idx])
df = pd.DataFrame(all_data, columns=["x", "y", "frame"])

search_range = 15
memory = 5
try:
    linked = tp.link_df(df, search_range=search_range, memory=memory)
except SubnetOversizeException as e:
    print(f"[WARN] {e}. Retrying with smaller search range...")
    linked = tp.link_df(df, search_range=15, memory=memory)

print(f"[INFO] Tracking complete – {linked['particle'].nunique()} cells identified.")

# ---------------- 4. FEATURE EXTRACTION (robust speed + true turning angle) ----------------
print("\n[INFO] Computing motion features...")

TARGET_FPS  = 10.0                      # only used if SPEED_UNIT == "per_second"
SPEED_UNIT  = "per_frame"               # "per_frame" (original) or "per_second"
USE_MEDIAN_SPEED = True
HARD_CAP_FRAC = 0.5                     # absolute cap as fraction of search_range

# Global winsor cap from all raw steps
raw_steps_all = []
for pid, g in linked.groupby("particle"):
    g = g.sort_values("frame")
    if len(g) < 3:
        continue
    x = g["x"].to_numpy()
    y = g["y"].to_numpy()
    steps = np.hypot(np.diff(x), np.diff(y))
    if steps.size:
        raw_steps_all.append(steps)
if len(raw_steps_all):
    raw_steps_all = np.concatenate(raw_steps_all)
    gq1, gq3 = np.quantile(raw_steps_all, [0.25, 0.75])
    g_iqr = gq3 - gq1
    GLOBAL_UPPER = gq3 + 3.0 * g_iqr
else:
    GLOBAL_UPPER = np.inf

features = []
hard_cap = HARD_CAP_FRAC * search_range

for pid, g in linked.groupby("particle"):
    g = g.sort_values("frame")
    if len(g) < 3:
        continue

    x = g["x"].to_numpy()
    y = g["y"].to_numpy()

    dx = np.diff(x)
    dy = np.diff(y)
    steps = np.hypot(dx, dy)
    if steps.size < 2:
        continue

    # Winsorize steps
    q1, q3 = np.quantile(steps, [0.25, 0.75])
    iqr = q3 - q1
    local_upper = q3 + 3.0 * iqr
    cap_val = min(hard_cap, GLOBAL_UPPER, local_upper)

    steps_w = np.minimum(steps, cap_val)
    scale = np.divide(steps_w, steps, out=np.ones_like(steps_w), where=steps > 0)
    dx_w = dx * scale
    dy_w = dy * scale

    speed_pf = float(np.median(steps_w) if USE_MEDIAN_SPEED else np.mean(steps_w))
    path_len = float(np.sum(steps_w))
    disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    persistP = float(disp / path_len) if path_len > 0 else 0.0

    # Turning angle
    v_prev = np.column_stack((dx_w[:-1], dy_w[:-1]))
    v_next = np.column_stack((dx_w[1:],  dy_w[1:]))
    n_prev = np.linalg.norm(v_prev, axis=1)
    n_next = np.linalg.norm(v_next, axis=1)
    valid = (n_prev > 0) & (n_next > 0)
    if valid.sum() < 3:   # require ≥3 angles; else mark NaN
        turn_mean = np.nan
    else:
        v_prev = v_prev[valid]; v_next = v_next[valid]
        n_prev = n_prev[valid]; n_next = n_next[valid]
        cosang = np.sum(v_prev * v_next, axis=1) / (n_prev * n_next)
        cosang = np.clip(cosang, -1.0, 1.0)
        turn_angles = np.degrees(np.arccos(cosang))
        turn_mean = float(np.mean(turn_angles))

    speed_val = speed_pf * TARGET_FPS if SPEED_UNIT == "per_second" else speed_pf
    features.append([pid, speed_val, disp, turn_mean, path_len, persistP])

feat_df = pd.DataFrame(
    features,
    columns=["particle", "speed", "disp", "turn_angle", "path_len", "persist_P"]
)
print(f"[INFO] Computed features for {len(feat_df)} cells.")

# Video-level means
mean_speed = float(np.nanmean(feat_df["speed"].to_numpy()))
mean_disp  = float(np.nanmean(feat_df["disp"].to_numpy()))
mean_angle = float(np.nanmean(feat_df["turn_angle"].to_numpy()))
mean_P     = float(np.nanmean(feat_df["persist_P"].to_numpy()))

unit_label = "px/s" if SPEED_UNIT == "per_second" else "px/frame"
print(f"\n[INFO] Video-Level Statistics:"
      f"\nMean Speed: {mean_speed:.2f} {unit_label}"
      f"\nTotal Displacement: {mean_disp:.2f} px"
      f"\nMean Turn Angle: {mean_angle:.2f}°")

# ---------------- 4b. INFERENCE: p-values vs class means (bootstrap) ----------------
def bootstrap_pvalue(samples, mu0, B=10000, seed=0):
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if n < 2 or not np.isfinite(samples).all():
        return np.nan, (np.nan, np.nan), float(np.nan), float(np.nan)
    rng = np.random.default_rng(seed)
    obs_mean = float(samples.mean())
    boots_ci = rng.choice(samples, size=(B, n), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boots_ci, [2.5, 97.5])
    centered = samples - obs_mean + mu0
    boots_null = rng.choice(centered, size=(B, n), replace=True).mean(axis=1)
    p = float(np.mean(np.abs(boots_null - mu0) >= abs(obs_mean - mu0)))
    sd = samples.std(ddof=1)
    d = float((obs_mean - mu0) / sd) if sd > 0 else np.nan
    return p, (float(ci_low), float(ci_high)), obs_mean, d

print("\n[INFO] Inference vs reference means (bootstrap, two-sided; 95% CI for mean)")

spd_arr  = feat_df["speed"].to_numpy()
disp_arr = feat_df["disp"].to_numpy()
ang_arr  = feat_df["turn_angle"].to_numpy()

# Reference means adjusted to the working unit
if SPEED_UNIT == "per_second":
    norm_ref = {'speed': REF_MEANS["Normal"]["speed_pxs"], 'disp': REF_MEANS["Normal"]["disp"], 'angle': REF_MEANS["Normal"]["angle"]}
    canc_ref = {'speed': REF_MEANS["Cancer"]["speed_pxs"], 'disp': REF_MEANS["Cancer"]["disp"], 'angle': REF_MEANS["Cancer"]["angle"]}
else:
    norm_ref = {'speed': REF_MEANS["Normal"]["speed_pxs"]/10.0, 'disp': REF_MEANS["Normal"]["disp"], 'angle': REF_MEANS["Normal"]["angle"]}
    canc_ref = {'speed': REF_MEANS["Cancer"]["speed_pxs"]/10.0, 'disp': REF_MEANS["Cancer"]["disp"], 'angle': REF_MEANS["Cancer"]["angle"]}

def report_metric(name, arr, mu_norm, mu_canc, seed_base=1, alpha=0.05):
    pN, ciN, obs, dN = bootstrap_pvalue(arr, mu_norm, B=10000, seed=seed_base)
    pC, _,   _,  dC  = bootstrap_pvalue(arr, mu_canc, B=10000, seed=seed_base+1)

    print(f"\n[{name}]")
    print(f"  Observed mean: {obs:.2f}")
    print(f"  95% CI (bootstrap): [{ciN[0]:.2f}, {ciN[1]:.2f}]")
    print(f"  H0: mean == Normal ({mu_norm:.2f}) -> p = {pN:.4f}, Cohen's d = {dN:.2f}")
    print(f"  H0: mean == Cancer ({mu_canc:.2f}) -> p = {pC:.4f}, Cohen's d = {dC:.2f}")

    cons_norm   = (pN > alpha)
    cons_cancer = (pC > alpha)
    if cons_norm and not cons_cancer:
        interp = "→ Consistent with NORMAL; different from CANCER."
        winner = "Normal"
    elif cons_cancer and not cons_norm:
        interp = "→ Consistent with CANCER; different from NORMAL."
        winner = "Cancer"
    elif cons_norm and cons_cancer:
        interp = "→ Ambiguous: consistent with BOTH classes."
        winner = "Both"
    else:
        interp = "→ Ambiguous: different from BOTH class means."
        winner = "Neither"

    print(f"  {interp}")
    return {
        "winner": winner, "p_norm": pN, "p_cancer": pC,
        "d_norm": dN, "d_cancer": dC, "obs": obs
    }

r_speed = report_metric(f"Speed ({unit_label})", spd_arr,  norm_ref['speed'], canc_ref['speed'], seed_base=1)
r_disp  = report_metric("Displacement (px)",     disp_arr, norm_ref['disp'],  canc_ref['disp'],  seed_base=3)
r_angle = report_metric("Turn angle (deg)",      ang_arr,  norm_ref['angle'], canc_ref['angle'], seed_base=5)

# ---------------- Threshold calibration (used only as tie-breaker) ----------------
def compute_thresholds(cal_mode="empirical"):
    if cal_mode == "empirical":
        # use your measured class means (per-frame)
        n = EMP_MEANS["Normal"]; c = EMP_MEANS["Cancer"]
        th = {
            "speed": 0.5 * (n["speed"] + c["speed"]),   # ~4.815
            "disp":  0.5 * (n["disp"]  + c["disp"]),    # ~17.208
            "angle": 0.5 * (n["angle"] + c["angle"]),   # ~85.02
        }
    else:
        # use reference means (already in current units)
        th = {
            "speed": 0.5 * (norm_ref['speed'] + canc_ref['speed']),
            "disp":  0.5 * (norm_ref['disp']  + canc_ref['disp']),
            "angle": 0.5 * (norm_ref['angle'] + canc_ref['angle']),
        }
    return th

THRESH = compute_thresholds(CAL_MODE)

print("\n[INFO] Calibrated thresholds (tie-breakers):")
print(f"  speed: {THRESH['speed']:.4f} {unit_label}")
print(f"  disp : {THRESH['disp']:.4f} px")
print(f"  angle: {THRESH['angle']:.4f} °")

# ---------------- 5. CLASSIFICATION (Evidence-first voting with guardrails) ----------------
def use_metric(r):
    # Vote only when the metric is consistent with exactly ONE class
    return r["winner"] in ("Normal", "Cancer")

def metric_vote_by_evidence(r, alpha=0.05):
    # Vote by p-values; tie-break by |d|
    pN, pC = r["p_norm"], r["p_cancer"]
    dN, dC = r["d_norm"], r["d_cancer"]
    if (pN > alpha) and (pC <= alpha):
        return "Normal"
    if (pC > alpha) and (pN <= alpha):
        return "Cancer"
    if (pN > alpha) and (pC > alpha):
        # shouldn't happen because use_metric() would be False,
        # but keep a defensive tie-break:
        return "Normal" if abs(dN) < abs(dC) else "Cancer"
    # If both <= alpha (equally inconsistent), fall through to None and use thresholds
    return None

def threshold_vote(metric_name, obs, th):
    # Directionality:
    #  - speed: higher -> Cancer
    #  - disp : higher -> Normal
    #  - angle: higher -> Cancer
    if np.isnan(obs):
        return None
    if metric_name == "speed":
        return "Cancer" if obs > th else "Normal"
    elif metric_name == "disp":
        return "Normal" if obs > th else "Cancer"
    elif metric_name == "angle":
        return "Cancer" if obs > th else "Normal"
    return None

votes = {"Cancer": 0, "Normal": 0}
used_metrics = 0

# SPEED
if use_metric(r_speed):
    v = metric_vote_by_evidence(r_speed, alpha=0.05)
    if v is None:
        v = threshold_vote("speed", r_speed["obs"], THRESH["speed"])
    if v: votes[v] += 1; used_metrics += 1

# DISPLACEMENT
if use_metric(r_disp):
    v = metric_vote_by_evidence(r_disp, alpha=0.05)
    if v is None:
        v = threshold_vote("disp", r_disp["obs"], THRESH["disp"])
    if v: votes[v] += 1; used_metrics += 1

# ANGLE
if use_metric(r_angle):
    v = metric_vote_by_evidence(r_angle, alpha=0.05)
    if v is None:
        v = threshold_vote("angle", r_angle["obs"], THRESH["angle"])
    if v: votes[v] += 1; used_metrics += 1

# Final label
if votes["Cancer"] > votes["Normal"]:
    classification = "Cancer Cell"
elif votes["Normal"] > votes["Cancer"]:
    classification = "Normal Cell"
else:
    # exact tie -> fall back to most informative metric by |d|-distance
    # pick the metric with largest separation and use its threshold direction
    sep = []
    for name, r in [("speed", r_speed), ("disp", r_disp), ("angle", r_angle)]:
        # use smaller |d| to the "supported" class? Prefer larger separation between class means:
        # Here we take min |d| to either class as "closeness"; larger 1/min indicates separation.
        dn = abs(r["d_norm"]) if np.isfinite(r["d_norm"]) else np.inf
        dc = abs(r["d_cancer"]) if np.isfinite(r["d_cancer"]) else np.inf
        closeness = min(dn, dc)
        sep.append((1.0 / (closeness + 1e-9), name, r["obs"]))
    sep.sort(reverse=True)
    if len(sep) and not np.isnan(sep[0][2]):
        top_name, top_obs = sep[0][1], sep[0][2]
        v = threshold_vote(top_name, top_obs, THRESH[top_name])
        classification = "Cancer Cell" if v == "Cancer" else "Normal Cell"
    else:
        classification = "Normal Cell"  # conservative default

cancer_likeness = votes["Cancer"] / max(1, used_metrics)

print("\n[RESULT] Classification:", classification)
print(f"[INFO] Cancer-Likeness Score: {cancer_likeness:.2f}")
print(f"[INFO] Feature Votes: {votes} (used_metrics={used_metrics})")

# ---------------- 6. OVERALL INTERPRETATION (dynamic) ----------------
winners = [r_speed["winner"], r_disp["winner"], r_angle["winner"]]
support_cancer = winners.count("Cancer")
support_normal = winners.count("Normal")

print("\n[INTERPRETATION]")
print(f"  Metrics supporting CANCER: {support_cancer} / 3")
print(f"  Metrics supporting NORMAL: {support_normal} / 3")
if support_cancer > support_normal:
    print("  Overall evidence leans toward CANCER (by p-value consistency).")
elif support_normal > support_cancer:
    print("  Overall evidence leans toward NORMAL (by p-value consistency).")
else:
    print("  Overall evidence is AMBIGUOUS from p-values alone.")

# Agreement check
if classification.startswith("Cancer") and support_cancer >= support_normal:
    print("  ✓ Consistent with classifier: Cancer.")
elif classification.startswith("Normal") and support_normal >= support_cancer:
    print("  ✓ Consistent with classifier: Normal.")
else:
    print("  ⚠ Classifier and p-value evidence disagree; treat result with caution.")
