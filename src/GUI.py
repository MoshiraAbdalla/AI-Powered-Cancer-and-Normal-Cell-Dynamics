#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cell Movement Classifier — Pro UI (Colorful & Clinic-Friendly)
"""

import os
import sys
import time
import threading
import ctypes

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from skimage import io, filters, morphology, measure
import trackpy as tp
from trackpy.linking.utils import SubnetOversizeException

# Matplotlib (offscreen)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== Colors =====
COLOR_BG        = "#0f172a"
COLOR_TEXT      = "#e6edf7"
COLOR_HEADER_BG = "#0ea5a4"
COLOR_HEADER_TX = "#042f2e"
COLOR_PRIMARY   = "#3b82f6"
COLOR_ACCENT    = "#22c55e"
COLOR_DANGER    = "#ef4444"
COLOR_TAB_BG    = "#ecfeff"
COLOR_CARD      = "#ffffff"
COLOR_BORDER    = "#e2e8f0"

# ===== HiDPI =====
def enable_hidpi(root: tk.Tk, scaling=1.8):
    try:
        if sys.platform == "win32":
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        root.tk.call("tk", "scaling", scaling)
    except Exception:
        pass

# ===== UI image helpers =====
PREVIEW_SIZE = (1400, 900)
PINK_TINT_BGR = (180, 105, 255)

def to_tk(img_bgr_or_gray, size=None):
    if img_bgr_or_gray is None:
        return None
    if len(img_bgr_or_gray.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
    if size is None:
        size = PREVIEW_SIZE
    if size:
        img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(img_rgb))

def build_segmentation_overlay(gray_path, mask_path):
    gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gray is None or mask is None:
        return None
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tint = base.copy()
    tint[mask > 0] = PINK_TINT_BGR
    return cv2.addWeighted(base, 0.6, tint, 0.4, 0)

def overlay_edges(gray_path, mask_path, out_size=None):
    try:
        blend = build_segmentation_overlay(gray_path, mask_path)
        if blend is None:
            return None
        if out_size is None:
            out_size = PREVIEW_SIZE
        if out_size:
            blend = cv2.resize(blend, out_size, interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)))
    except Exception:
        return None

# ===== Tracking overlay =====
_PALETTE_RGB = [
    (31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),
    (140,86,75),(227,119,194),(127,127,127),(188,189,34),(23,190,207),
    (255,152,150),(197,176,213),(196,156,148),(247,182,210),(199,199,199),
    (219,219,141),(158,218,229),(57,59,121),(82,84,163),(107,110,207),
    (156,158,222),(99,121,57),(140,162,82),(181,207,107),(206,219,156),
    (140,109,49),(189,158,57),(231,186,82),(231,203,148),(132,60,57),
    (173,73,74),(214,97,107),(231,150,156),(123,65,115),(165,81,148),
    (206,109,189),(222,158,214),(57,59,121),(107,110,207),(99,121,57),
]
def _particle_color(pid):
    idx = (hash(str(pid)) & 0x7fffffff) % len(_PALETTE_RGB)
    r,g,b = _PALETTE_RGB[idx]
    return (b,g,r)

def build_tracking_overlay(base_frame_path, linked_df, tail_len=16, thickness=2, alpha=0.98):
    frame = cv2.imread(base_frame_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None
    base = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    if linked_df is None or "particle" not in linked_df.columns:
        return base
    for pid, group in linked_df.groupby("particle"):
        g = group.sort_values("frame")
        pts = np.round(g[["x","y"]].to_numpy()).astype(int)
        if len(pts) < 2:
            continue
        pts = pts[-(tail_len+1):]
        color = _particle_color(pid)
        for i in range(1, len(pts)):
            cv2.line(overlay, tuple(pts[i-1]), tuple(pts[i]), color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(overlay, tuple(pts[-1]), max(2, thickness), color, -1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

def draw_tracks_overlay(base_frame_path, linked_df, size=None):
    try:
        img = build_tracking_overlay(base_frame_path, linked_df, tail_len=16, thickness=1, alpha=0.98)
    except Exception:
        return None
    if img is None:
        return None
    if size is None:
        size = PREVIEW_SIZE
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

# ===== Trajectories plot =====
def make_trajectories_plot(linked_df, frame_size, title="Cell Movement", out_path=None):
    if linked_df is None or linked_df.empty or "particle" not in linked_df.columns:
        return None
    w,h = frame_size
    fig = plt.figure(figsize=(10.5, 8.5), dpi=180)
    ax = fig.add_subplot(111)
    for pid, g in linked_df.groupby("particle"):
        g = g.sort_values("frame")
        x = g["x"].values
        y = g["y"].values
        if len(x) < 2:
            continue
        idx = (hash(str(pid)) & 0x7fffffff) % len(_PALETTE_RGB)
        r,gc,b = _PALETTE_RGB[idx]
        color = (r/255, gc/255, b/255)
        ax.plot(x, y, linewidth=1.2, color=color)
        step = max(1, len(x)//10)
        for i in range(0, len(x)-1, step):
            dx = x[i+1]-x[i]; dy = y[i+1]-y[i]
            ax.arrow(x[i], y[i], dx, dy, shape='full', lw=0, length_includes_head=True,
                     head_width=3, head_length=6, color=color, alpha=0.85)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_xlabel("X position (pixels)"); ax.set_ylabel("Y position (pixels)")
    ax.set_title(title); ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight"); plt.close(fig); return out_path
    return fig

# ===== Pipeline =====
def run_pipeline(video_path, results_root, log_fn=print, progress_fn=lambda *_: None, preview_fn=lambda *_: None):
    run_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(results_root, run_name)
    frames_dir = os.path.join(out_dir, "frames")
    seg_dir = os.path.join(out_dir, "segmentation")
    os.makedirs(frames_dir, exist_ok=True); os.makedirs(seg_dir, exist_ok=True)

    # 1) Frames
    progress_fn(1, "Extracting frames…")
    log_fn("\n[INFO] Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log_fn(f"[INFO] Video Metadata:\nFrames: {frame_count}, FPS: {fps}, Resolution: {(fw, fh)}")

    i = 0; PREV_EVERY = 10
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fpath = os.path.join(frames_dir, f"frame_{i:03d}.png")
        io.imsave(fpath, gray)
        if i % PREV_EVERY == 0:
            preview_fn({"type": "pre_frame", "path": fpath})
        if frame_count:
            frac = min(0.99, (i + 1) / frame_count)
            preview_fn({"type": "progress", "base": 0, "frac": frac,
                        "msg": f"Extracting frames… ({i+1}/{frame_count})"})
        i += 1
    cap.release()
    log_fn(f"[INFO] Saved {i} frames to '{frames_dir}'")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

    # 2) Segmentation
    progress_fn(2, "Segmenting cells…")
    log_fn("\n[INFO] Performing segmentation...")
    SEG_PREV_EVERY = 10
    for idx, f in enumerate(frame_files):
        img = io.imread(os.path.join(frames_dir, f))
        thr = filters.threshold_otsu(img)
        mask = img > thr
        mask = morphology.remove_small_objects(mask, 40)
        mask = morphology.remove_small_holes(mask, 40)
        mpath = os.path.join(seg_dir, f)
        io.imsave(mpath, (mask * 255).astype(np.uint8))
        if idx % SEG_PREV_EVERY == 0:
            preview_fn({"type": "seg_overlay",
                        "frame_path": os.path.join(frames_dir, f),
                        "mask_path": mpath})
        total = max(1, len(frame_files))
        frac = (idx + 1) / total
        preview_fn({"type": "progress", "base": 1, "frac": frac,
                    "msg": f"Segmenting cells… ({idx+1}/{total})"})
    log_fn(f"[INFO] Segmentation complete – saved masks to '{seg_dir}'")
    mask_files = sorted([f for f in os.listdir(seg_dir) if f.endswith(".png")])

    # 3) Tracking
    progress_fn(3, "Tracking cells…")
    log_fn("\n[INFO] Tracking cells...")
    all_data = []
    for idx, f in enumerate(mask_files):
        img = io.imread(os.path.join(seg_dir, f))
        labeled = measure.label(img > 0)
        for p in measure.regionprops(labeled):
            y,x = p.centroid
            all_data.append([x, y, idx])
    df = pd.DataFrame(all_data, columns=["x","y","frame"])

    search_range = 25; memory = 5
    try:
        linked = tp.link_df(df, search_range=search_range, memory=memory)
    except SubnetOversizeException as e:
        log_fn(f"[WARN] {e}. Retrying with smaller search range…")
        linked = tp.link_df(df, search_range=15, memory=memory)

    cells = int(linked["particle"].nunique()) if "particle" in linked.columns else 0
    log_fn(f"[INFO] Tracking complete – {cells} cells identified.")

    # Save tracks
    os.makedirs(out_dir, exist_ok=True)
    tracks_csv = os.path.join(out_dir, "tracks_linked.csv")
    try:
        linked.to_csv(tracks_csv, index=False); log_fn(f"[INFO] Saved tracks: {tracks_csv}")
    except Exception as e:
        log_fn(f"[WARN] Could not save tracks CSV: {e}")

    # Overlay + plot
    if frame_files:
        last_frame_path = os.path.join(frames_dir, frame_files[-1])
        overlay_img = build_tracking_overlay(last_frame_path, linked, tail_len=16, thickness=2, alpha=0.98)
        if overlay_img is not None:
            ov_path = os.path.join(out_dir, "tracking_overlay_last.png")
            try: cv2.imwrite(ov_path, overlay_img); log_fn(f"[INFO] Saved overlay: {ov_path}")
            except Exception as e: log_fn(f"[WARN] Could not save overlay image: {e}")
        preview_fn({"type":"tracking_overlay","frame_path":last_frame_path,"linked_df":linked})
    preview_fn({"type":"progress","base":2,"frac":1.0,"msg":"Tracking cells… done"})

    if frame_files:
        last_gray = cv2.imread(os.path.join(frames_dir, frame_files[-1]), cv2.IMREAD_GRAYSCALE)
        fh, fw = last_gray.shape[:2]
        traj_png = os.path.join(out_dir, "trajectories_plot.png")
        title = "Cancer Cell Movement" if "cancer" in run_name.lower() else "Cell Movement"
        try:
            make_trajectories_plot(linked, (fw, fh), title=title, out_path=traj_png)
            preview_fn({"type":"traj_plot","path":traj_png})
            log_fn(f"[INFO] Saved trajectories plot: {traj_png}")
        except Exception as e:
            log_fn(f"[WARN] Could not create trajectories plot: {e}")

    # 4) Features — EXACTLY like classify_new_video.py
    preview_fn({"type":"progress","base":3,"frac":0.2,"msg":"Computing features…"})
    log_fn("\n[INFO] Computing motion features...")
    features = []
    for pid, group in linked.groupby("particle"):
        if len(group) < 3: continue
        group = group.sort_values("frame")
        dx = np.diff(group["x"].to_numpy())
        dy = np.diff(group["y"].to_numpy())
        speed = float(np.mean(np.sqrt(dx**2 + dy**2)))  # px/frame (reported as px/s in script)
        disp  = float(np.sqrt((group["x"].iloc[-1] - group["x"].iloc[0])**2 +
                              (group["y"].iloc[-1] - group["y"].iloc[0])**2))
        ang   = float(np.mean(np.abs(np.degrees(np.arctan2(dy, dx)))))


        # dx = np.diff(g["x"]); dy = np.diff(g["y"])
        step = np.sqrt(dx**2 + dy**2)

        path_len = float(np.sum(step))                                    # L
        # disp     = float(np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2))      # D
        persistP = float(disp / path_len) if path_len > 0 else 0.0        # P = D/L

        # speed    = float(np.mean(step))                                   # px/frame (reported as px/s)
        # ang      = float(np.mean(np.abs(np.degrees(np.arctan2(dy, dx)))))

        features.append([pid, speed, disp, ang, path_len, persistP])      # ADD path_len & persistP





        # features.append([pid, speed, disp, ang])
    # feat_df = pd.DataFrame(features, columns=["particle","speed","disp","angle"])
    feat_df = pd.DataFrame(
    features,
    columns=["particle", "speed", "disp", "angle", "path_len", "persist_P"]  # ADD columns
)

    preview_fn({"type":"progress","base":3,"frac":1.0,"msg":"Computing features… done"})
    log_fn(f"[INFO] Computed features for {len(feat_df)} cells.")

    feat_csv = os.path.join(out_dir, "features.csv")
    try: feat_df.to_csv(feat_csv, index=False); log_fn(f"[INFO] Saved features: {feat_csv}")
    except Exception as e: log_fn(f"[WARN] Could not save features CSV: {e}")

    if len(feat_df) == 0:
        mean_speed = mean_disp = mean_angle = 0.0
    else:
        mean_speed = float(feat_df["speed"].mean())   # same as script
        mean_disp  = float(feat_df["disp"].mean())
        mean_angle = float(feat_df["angle"].mean())
        mean_P = float(feat_df["persist_P"].mean()) if len(feat_df) else 0.0  # ADD

    # Script-style logging
    log_fn(
        f"\n[INFO] Video-Level Statistics:"
        f"\nMean Speed: {mean_speed:.2f} px/s"
        f"\nTotal Displacement: {mean_disp:.2f} px"
        f"\nMean Turn Angle: {mean_angle:.2f}°"
        f"\nMean Persistence (D/L): {mean_P:.3f}"   # ADD

    )

    # 5) Classification — same thresholds/logic as script
    preview_fn({"type":"progress","base":4,"frac":0.4,"msg":"Classifying…"})
    norm = {'speed': 35.96, 'disp': 26.08, 'angle': 69.7}
    canc = {'speed': 61.95, 'disp': 15.48, 'angle': 90.89}
    th_speed = (norm['speed'] + canc['speed']) / 2
    th_disp  = (norm['disp']  + canc['disp'])  / 2
    th_angle = (norm['angle'] + canc['angle']) / 2

    votes = {"Cancer": 0, "Normal": 0}
    votes["Cancer" if mean_speed > th_speed else "Normal"] += 1
    votes["Normal" if mean_disp > th_disp else "Cancer"] += 1
    votes["Cancer" if mean_angle > th_angle else "Normal"] += 1
    classification = "Cancer Cell" if votes["Cancer"] > votes["Normal"] else "Normal Cell"
    cancer_likeness = votes["Cancer"] / 3.0
    preview_fn({"type":"progress","base":4,"frac":1.0,"msg":"Classifying… done"})

    log_fn(f"\n[RESULT] Classification: {classification}")
    log_fn(f"[INFO] Cancer-Likeness Score: {cancer_likeness:.2f}")
    log_fn(f"[INFO] Feature Votes: {votes}")

    previews = {
        "frames_dir": frames_dir,
        "seg_dir": seg_dir,
        "frame_files": frame_files,
        "mask_files": mask_files,
        "linked_df": linked,
        "out_dir": out_dir,
        "units": "px/s",
        "mean_P": mean_P,   # ADD

    }
    # return EXACT means so GUI shows the same numbers
    return classification, votes, (mean_speed, mean_disp, mean_angle), previews

# ===== GUI =====
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cell  Classifier — Pro")
        try: self.state("zoomed")
        except Exception: self.geometry("1800x1100")
        self.minsize(1500, 900)

        enable_hidpi(self, scaling=1.8)
        self.style_init()

        self.results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
        os.makedirs(self.results_root, exist_ok=True)

        self.video_path = None
        self.video_var = tk.StringVar(value="No video selected")
        self.pred_var  = tk.StringVar(value="—")
        self.vote_var  = tk.StringVar(value="Cancer: 0 | Normal: 0")
        self.step_var  = tk.StringVar(value="Ready")

        self.mean_speed_var = tk.StringVar(value="—")
        self.mean_disp_var  = tk.StringVar(value="—")
        self.mean_angle_var = tk.StringVar(value="—")
        self.persist_var = tk.StringVar(value="—")   # ADD

        self.img_pre = None; self.img_seg = None; self.img_track = None; self.img_trajplot = None
        self._live_running = False; self._live_thread = None; self._live_writer = None; self._live_cap = None; self._live_path = None
        self.build_ui()

    def style_init(self):
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except Exception: pass
        style.configure(".", font=("Segoe UI", 14))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 18), foreground=COLOR_HEADER_TX)
        style.configure("Pred.Big.TLabel", font=("Segoe UI Semibold", 28))
        style.configure("Status.TLabel", foreground="#334155", font=("Segoe UI", 12))
        style.configure("Primary.TButton", background=COLOR_PRIMARY, foreground="white", padding=(14, 8))
        style.map("Primary.TButton", background=[("active", "#2563eb")])
        style.configure("Accent.TButton", background=COLOR_ACCENT, foreground="white", padding=(14, 8))
        style.map("Accent.TButton", background=[("active", "#16a34a")])
        style.configure("Danger.TButton", background=COLOR_DANGER, foreground="white", padding=(14, 8))
        style.map("Danger.TButton", background=[("active", "#dc2626")])
        style.configure("Ghost.TButton", background="#ffffff", foreground="#0f172a", padding=(12, 6), borderwidth=1, relief="solid")
        style.map("Ghost.TButton", background=[("active", "#f1f5f9")])
        style.configure("TNotebook.Tab", padding=(18, 10), font=("Segoe UI", 14))
        style.configure("TNotebook", background=COLOR_TAB_BG, borderwidth=0)
        style.configure("blue.Horizontal.TProgressbar", troughcolor="#e9edf5", background=COLOR_PRIMARY, thickness=14)
        style.configure("StatLabel.TLabel", font=("Segoe UI", 13))
        style.configure("StatValue.TLabel", font=("Segoe UI Semibold", 16), foreground="#2563eb")

    def preview_cb(self, evt: dict):
        def _apply():
            typ = evt.get("type")
            if typ == "pre_frame":
                img = cv2.imread(evt["path"], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.img_pre = to_tk(img); self.lbl_pre.configure(image=self.img_pre, text=""); self.tabs.select(self.tab_pre)
            elif typ == "seg_overlay":
                ov = overlay_edges(evt["frame_path"], evt["mask_path"])
                if ov: self.img_seg = ov; self.lbl_seg.configure(image=self.img_seg, text=""); self.tabs.select(self.tab_seg)
            elif typ == "tracking_overlay":
                trk = draw_tracks_overlay(evt["frame_path"], evt["linked_df"])
                if trk: self.img_track = trk; self.lbl_track.configure(image=self.img_track, text=""); self.tabs.select(self.tab_track)
            elif typ == "traj_plot":
                p = evt.get("path"); img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is not None:
                    self.img_trajplot = to_tk(img); self.lbl_traj.configure(image=self.img_trajplot, text=""); self.tabs.select(self.tab_traj)
            elif typ == "log":
                self.log_write(evt.get("msg", ""))
            elif typ == "progress":
                base = evt.get("base", 0); frac = float(evt.get("frac", 0.0))
                self.progress["value"] = min(5, base + frac); msg = evt.get("msg")
                if msg: self.step_var.set(msg)
        self.after(0, _apply)

    def build_ui(self):
        pad = 16
        header_wrap = tk.Frame(self, bg=COLOR_HEADER_BG); header_wrap.pack(fill="x")
        header = ttk.Frame(header_wrap, padding=(pad, pad, pad, pad)); header.pack(fill="x")
        ttk.Label(header, text="Cell  Classifier", style="Title.TLabel").pack(side="left")
        actions = ttk.Frame(header); actions.pack(side="right")
        self.btn_live_stop = ttk.Button(actions, text="Stop Live ■", style="Danger.TButton", command=self.stop_live, state="disabled"); self.btn_live_stop.pack(side="right", padx=(10,0))
        self.btn_live_start = ttk.Button(actions, text="Start Live ▶", style="Accent.TButton", command=self.start_live); self.btn_live_start.pack(side="right", padx=(10,0))
        ttk.Button(actions, text="Run Classification ▶", style="Primary.TButton", command=self.run_threaded).pack(side="right", padx=(10,0))
        ttk.Button(actions, text="Open Results Folder", style="Ghost.TButton", command=self.open_results).pack(side="right", padx=(10,0))

        src_card = tk.Frame(self, bg=COLOR_CARD, highlightbackground=COLOR_BORDER, highlightthickness=1); src_card.pack(fill="x", padx=pad, pady=(pad,8))
        src = ttk.Frame(src_card, padding=(pad, pad, pad, pad)); src.pack(fill="x")
        ttk.Label(src, text="Source:").pack(side="left", padx=(0,8))
        self.entry = ttk.Entry(src, textvariable=self.video_var, state="readonly"); self.entry.pack(side="left", fill="x", expand=True, ipady=6)
        ttk.Button(src, text="Open Video…", style="Ghost.TButton", command=self.choose_video).pack(side="left", padx=(10,0))

        prog_card = tk.Frame(self, bg=COLOR_CARD, highlightbackground=COLOR_BORDER, highlightthickness=1); prog_card.pack(fill="x", padx=pad, pady=(0,pad))
        prog = ttk.Frame(prog_card, padding=(pad,12,pad,12)); prog.pack(fill="x")
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=5, value=0, style="blue.Horizontal.TProgressbar"); self.progress.pack(fill="x")
        ttk.Label(prog, textvariable=self.step_var, style="Status.TLabel").pack(anchor="w", pady=(6,0))

        body = ttk.Panedwindow(self, orient="vertical"); body.pack(fill="both", expand=True, padx=pad, pady=(0,pad))
        self.tabs = ttk.Notebook(body); body.add(self.tabs, weight=8)

        self.tab_pre = ttk.Frame(self.tabs, padding=pad); self.tabs.add(self.tab_pre, text="1) Preprocess")
        ttk.Label(self.tab_pre, text="Preprocessed Frame", style="Title.TLabel").pack(anchor="w", pady=(0,8))
        self.lbl_pre = ttk.Label(self.tab_pre, text="(Live preview during run)", anchor="center"); self.lbl_pre.pack(fill="both", expand=True, padx=24, pady=12)

        self.tab_seg = ttk.Frame(self.tabs, padding=pad); self.tabs.add(self.tab_seg, text="2) Segmentation")
        ttk.Label(self.tab_seg, text="Segmentation Overlay", style="Title.TLabel").pack(anchor="w", pady=(0,8))
        self.lbl_seg = ttk.Label(self.tab_seg, text="(Run to see segmentation overlay)", anchor="center"); self.lbl_seg.pack(fill="both", expand=True, padx=24, pady=12)

        self.tab_track = ttk.Frame(self.tabs, padding=pad); self.tabs.add(self.tab_track, text="3) Tracking (Overlay)")
        ttk.Label(self.tab_track, text="Trajectory Overlay", style="Title.TLabel").pack(anchor="w", pady=(0,8))
        self.lbl_track = ttk.Label(self.tab_track, text="(Run to see trajectories)", anchor="center"); self.lbl_track.pack(fill="both", expand=True, padx=24, pady=12)

        self.tab_traj = ttk.Frame(self.tabs, padding=pad); self.tabs.add(self.tab_traj, text="4) Trajectories Plot")
        ttk.Label(self.tab_traj, text="Trajectories Direction Plot", style="Title.TLabel").pack(anchor="w", pady=(0,8))
        self.lbl_traj = ttk.Label(self.tab_traj, text="(Run to see trajectories plot)", anchor="center"); self.lbl_traj.pack(fill="both", expand=True, padx=24, pady=12)

        self.tab_cls = ttk.Frame(self.tabs, padding=pad); self.tabs.add(self.tab_cls, text="5) Classification")
        ttk.Label(self.tab_cls, text="Prediction", style="Title.TLabel").pack(anchor="w", pady=(0,8))
        self.lbl_pred = ttk.Label(self.tab_cls, textvariable=self.pred_var, style="Pred.Big.TLabel"); self.lbl_pred.pack(anchor="w", pady=(2,6))
        ttk.Label(self.tab_cls, text="Votes").pack(anchor="w")
        ttk.Label(self.tab_cls, textvariable=self.vote_var).pack(anchor="w", pady=(0,8))

        ttk.Separator(self.tab_cls, orient="horizontal").pack(fill="x", pady=(10,10))
        stats = ttk.Frame(self.tab_cls); stats.pack(anchor="w")
        ttk.Label(stats, text="Video-level Metrics", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,8))
        ttk.Label(stats, text="Mean Speed:", style="StatLabel.TLabel").grid(row=1, column=0, sticky="w", padx=(0,12))
        ttk.Label(stats, textvariable=self.mean_speed_var, style="StatValue.TLabel").grid(row=1, column=1, sticky="w")
        ttk.Label(stats, text="Total Displacement:", style="StatLabel.TLabel").grid(row=2, column=0, sticky="w", padx=(0,12))
        ttk.Label(stats, textvariable=self.mean_disp_var, style="StatValue.TLabel").grid(row=2, column=1, sticky="w")
        ttk.Label(stats, text="Mean Turn Angle:", style="StatLabel.TLabel").grid(row=3, column=0, sticky="w", padx=(0,12))
        ttk.Label(stats, textvariable=self.mean_angle_var, style="StatValue.TLabel").grid(row=3, column=1, sticky="w")

# ADD these two lines:
        ttk.Label(stats, text="Mean Persistence (D/L):", style="StatLabel.TLabel").grid(row=4, column=0, sticky="w", padx=(0, 12))
        ttk.Label(stats, textvariable=self.persist_var, style="StatValue.TLabel").grid(row=4, column=1, sticky="w")

        log_card = tk.Frame(self, bg=COLOR_BG, highlightbackground=COLOR_BORDER, highlightthickness=1); body.add(log_card, weight=2)
        logframe = ttk.Frame(log_card, padding=(pad,8,pad,8)); logframe.pack(fill="both", expand=True)
        ttk.Label(logframe, text="Run Log", style="Title.TLabel").pack(anchor="w", pady=(8,6))
        self.log = tk.Text(logframe, height=8, wrap="word", state="disabled", background=COLOR_BG, foreground=COLOR_TEXT, insertbackground="#ffffff", relief="flat")
        self.log.pack(fill="both", expand=True)

        self.update_idletasks()
        try: body.sashpos(0, int(self.winfo_height()*0.78))
        except Exception: pass

        menubar = tk.Menu(self)
        runmenu = tk.Menu(menubar, tearoff=0)
        runmenu.add_command(label="Run Classification ▶", command=self.run_threaded, accelerator="Ctrl+R")
        runmenu.add_command(label="Start Live ▶", command=self.start_live, accelerator="Ctrl+L")
        runmenu.add_command(label="Stop Live ■", command=self.stop_live, accelerator="Ctrl+Shift+L")
        menubar.add_cascade(label="Run", menu=runmenu); self.config(menu=menubar)

        self.bind_all("<Control-r>", lambda e: self.run_threaded())
        self.bind_all("<Control-l>", lambda e: self.start_live())
        self.bind_all("<Control-L>", lambda e: self.start_live())
        self.bind_all("<Control-Shift-L>", lambda e: self.stop_live())

        sw = self.winfo_screenwidth(); sh = self.winfo_screenheight()
        globals()["PREVIEW_SIZE"] = (max(1100, int(sw*0.78)), max(780, int(sh*0.66)))

    def log_write(self, s):
        self.log.configure(state="normal"); self.log.insert("end", s + "\n"); self.log.configure(state="disabled"); self.log.see("end")

    def choose_video(self):
        p = filedialog.askopenfilename(title="Choose a video file",
            filetypes=[("Video files","*.mp4;*.avi;*.mov;*.mkv"),("All files","*.*")])
        if not p: return
        self.video_path = p; self.video_var.set(p)

    def open_results(self):
        path = self.results_root
        try:
            if os.name == "nt": os.startfile(path)
            elif sys.platform == "darwin": os.system(f'open "{path}"')
            else: os.system(f'xdg-open "{path}"')
        except Exception:
            messagebox.showinfo("Results", path)

    def start_live(self):
        if self._live_running: return
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name=="nt" else cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera", "Could not open camera."); return
        live_dir = os.path.join(self.results_root, "Live"); os.makedirs(live_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S"); self._live_path = os.path.join(live_dir, f"live_{ts}.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        writer = cv2.VideoWriter(self._live_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h), True)
        if not writer.isOpened(): cap.release(); messagebox.showerror("Recorder","Could not create video writer."); return
        self._live_cap = cap; self._live_writer = writer; self._live_running = True
        self.btn_live_start.configure(state="disabled"); self.btn_live_stop.configure(state="normal")
        self.step_var.set("Recording live… (press Stop Live when ready)"); self.tabs.select(self.tab_pre)

        def loop():
            while self._live_running:
                ok, frame = cap.read()
                if not ok: break
                writer.write(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.img_pre = to_tk(gray); self.lbl_pre.configure(image=self.img_pre, text=""); self.update_idletasks()
            try: writer.release()
            except Exception: pass
            try: cap.release()
            except Exception: pass
            if self._live_path and os.path.exists(self._live_path):
                self.video_path = self._live_path; self.video_var.set(self._live_path)
                self.step_var.set("Live recording saved. Ready to run classification.")
            self.btn_live_start.configure(state="normal"); self.btn_live_stop.configure(state="disabled")
        self._live_thread = threading.Thread(target=loop, daemon=True); self._live_thread.start()

    def stop_live(self): 
        if not self._live_running: return
        self._live_running = False

    def run_threaded(self):
        if not self.video_path:
            messagebox.showwarning("No video", "Please choose a video or record live first."); return
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        try:
            self.progress.configure(value=0, maximum=5); self.step_var.set("Starting…")
            self.log.configure(state="normal"); self.log.delete("1.0","end"); self.log.configure(state="disabled")
            for lbl, text in [(self.lbl_seg,"(Run to see segmentation overlay)"),
                              (self.lbl_track,"(Run to see trajectories)"),
                              (self.lbl_traj,"(Run to see trajectories plot)")]:
                lbl.configure(image="", text=text)
            self.pred_var.set("—"); self.vote_var.set("Cancer: 0 | Normal: 0")
            self.mean_speed_var.set("—"); self.mean_disp_var.set("—"); self.mean_angle_var.set("—")

            def progress(step, msg):
                self.progress["value"] = step; self.step_var.set(msg); self.log_write(f"[STEP] {msg}")

            classification, votes, means, previews = run_pipeline(
                self.video_path, self.results_root,
                log_fn=self.log_write, progress_fn=progress, preview_fn=self.preview_cb,
            )

            self.pred_var.set(classification)
            self.lbl_pred.configure(foreground=(COLOR_DANGER if classification.startswith("Cancer") else COLOR_PRIMARY))
            self.vote_var.set(f"Cancer: {votes['Cancer']} | Normal: {votes['Normal']}")

            mean_speed, mean_disp, mean_angle = means
            self.mean_speed_var.set(f"{mean_speed:.2f} px/s")
            self.mean_disp_var.set(f"{mean_disp:.2f} px")
            self.mean_angle_var.set(f"{mean_angle:.2f}°")
            # ADD these two lines:
            mean_P = previews.get("mean_P", 0.0)
            self.persist_var.set(f"{mean_P:.3f}")

            self.progress["value"] = 5; self.step_var.set("Done"); self.tabs.select(self.tab_cls)
        except Exception as e:
            messagebox.showerror("Error", str(e)); self.log_write(f"[ERROR] {e}")

if __name__ == "__main__":
    App().mainloop()
