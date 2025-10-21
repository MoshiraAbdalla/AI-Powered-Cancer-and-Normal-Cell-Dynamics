#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from skimage import io, filters, morphology, measure
import trackpy as tp
from trackpy.linking.utils import SubnetOversizeException


# ---------- small display helpers ----------
def to_tk(img_bgr_or_gray, size=(520, 380)):
    if img_bgr_or_gray is None:
        return None
    if len(img_bgr_or_gray.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
    if size:
        img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(img_rgb))


def overlay_edges(gray_path, mask_path, out_size=(520, 380)):
    try:
        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            return None
        edges = cv2.Canny(mask, 50, 150)
        gray_col = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        over = gray_col.copy()
        over[edges > 0] = (0, 0, 255)
        blend = cv2.addWeighted(gray_col, 0.85, over, 0.15, 0)
        if out_size:
            blend = cv2.resize(blend, out_size, interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)))
    except Exception:
        return None


# ---------- pipeline using ONLY your logic ----------
def run_classification_pipeline(video_path, results_root, log_fn=print):
    """
    Runs the exact pipeline the user provided and returns:
    {
      'classification': str,
      'votes': {'Cancer': int, 'Normal': int},
      'means': (mean_speed, mean_disp, mean_angle),
      'dirs': {'frames': frames_dir, 'seg': seg_dir}
    }
    """
    # Make run folder
    run_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(results_root, run_name)
    frames_dir = os.path.join(out_dir, "frames")
    seg_dir = os.path.join(out_dir, "segmentation")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    # ---------------- 1) FRAME EXTRACTION ----------------
    log_fn("\n[INFO] Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    res = (int(cap.get(3)), int(cap.get(4)))
    log_fn(f"[INFO] Video Metadata:\nFrames: {frame_count}, FPS: {fps}, Resolution: {res}")

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        io.imsave(os.path.join(frames_dir, f"frame_{i:03d}.png"), gray)
        i += 1
    cap.release()
    log_fn(f"[INFO] Saved {i} frames to '{frames_dir}'")

    # ---------------- 2) SEGMENTATION ----------------
    log_fn("\n[INFO] Performing segmentation...")
    for f in sorted(os.listdir(frames_dir)):
        img = io.imread(os.path.join(frames_dir, f))
        thr = filters.threshold_otsu(img)
        mask = img > thr
        mask = morphology.remove_small_objects(mask, 40)
        mask = morphology.remove_small_holes(mask, 40)
        io.imsave(os.path.join(seg_dir, f), (mask * 255).astype(np.uint8))
    log_fn(f"[INFO] Segmentation complete – saved masks to '{seg_dir}'")

    # ---------------- 3) TRACKING ----------------
    log_fn("\n[INFO] Tracking cells...")
    all_data = []
    for idx, f in enumerate(sorted(os.listdir(seg_dir))):
        img = io.imread(os.path.join(seg_dir, f))
        labeled = measure.label(img > 0)
        props = measure.regionprops(labeled)
        for p in props:
            y, x = p.centroid
            all_data.append([x, y, idx])
    df = pd.DataFrame(all_data, columns=["x", "y", "frame"])

    search_range = 25
    memory = 5
    try:
        linked = tp.link_df(df, search_range=search_range, memory=memory)
    except SubnetOversizeException as e:
        log_fn(f"[WARN] {e}. Retrying with smaller search range...")
        linked = tp.link_df(df, search_range=15, memory=memory)

    track_count = int(linked["particle"].nunique()) if "particle" in linked.columns else 0
    log_fn(f"[INFO] Tracking complete – {track_count} cells identified.")

    # ---------------- 4) FEATURE EXTRACTION ----------------
    log_fn("\n[INFO] Computing motion features...")
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
    log_fn(f"[INFO] Computed features for {len(feat_df)} cells.")

    # If empty after filtering, avoid NaNs
    if len(feat_df) == 0:
        mean_speed = 0.0
        mean_disp = 0.0
        mean_angle = 0.0
    else:
        mean_speed = float(feat_df["speed"].mean())
        mean_disp = float(feat_df["disp"].mean())
        mean_angle = float(feat_df["angle"].mean())

    log_fn(f"\n[INFO] Video-Level Statistics:"
           f"\nMean Speed: {mean_speed:.2f} px/s"
           f"\nTotal Displacement: {mean_disp:.2f} px"
           f"\nMean Turn Angle: {mean_angle:.2f}°")

    # ---------------- 5) CLASSIFICATION (VOTING MODEL) ----------------
    norm = {'speed': 35.96, 'disp': 26.08, 'angle': 69.7}
    canc = {'speed': 61.95, 'disp': 15.48, 'angle': 90.89}

    th_speed = (norm['speed'] + canc['speed']) / 2
    th_disp  = (norm['disp']  + canc['disp'])  / 2
    th_angle = (norm['angle'] + canc['angle']) / 2

    votes = {"Cancer": 0, "Normal": 0}

    # Speed: higher = cancer
    votes["Cancer" if mean_speed > th_speed else "Normal"] += 1
    # Displacement: higher = normal
    votes["Normal" if mean_disp > th_disp else "Cancer"] += 1
    # Turn angle: higher = cancer
    votes["Cancer" if mean_angle > th_angle else "Normal"] += 1

    classification = "Cancer Cell" if votes["Cancer"] > votes["Normal"] else "Normal Cell"
    cancer_likeness = votes["Cancer"] / 3.0

    log_fn(f"\n[RESULT] Classification: {classification}")
    log_fn(f"[INFO] Cancer-Likeness Score: {cancer_likeness:.2f}")
    log_fn(f"[INFO] Feature Votes: {votes}")

    return {
        "classification": classification,
        "votes": votes,
        "means": (mean_speed, mean_disp, mean_angle),
        "dirs": {"frames": frames_dir, "seg": seg_dir}
    }


# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cell Movement Classifier — Stepwise (Single-file)")
        self.geometry("1180x760")
        self.minsize(1120, 720)

        self.results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
        os.makedirs(self.results_root, exist_ok=True)

        self.video_path = None
        self.video_var = tk.StringVar(value="No video selected")
        self.pred_var = tk.StringVar(value="—")
        self.vote_var = tk.StringVar(value="Cancer: 0 | Normal: 0")
        self.thresh_var = tk.StringVar(value="—")
        self.step_var = tk.StringVar(value="Ready")

        # keep images alive
        self.img_pre = None
        self.img_seg = None

        self.build_ui()

    def build_ui(self):
        pad = 10

        # top area (two rows)
        top = ttk.Frame(self); top.pack(fill="x", padx=pad, pady=(pad, 4))
        row0 = ttk.Frame(top); row0.pack(fill="x", pady=(0,6))
        ttk.Label(row0, text="Selected:").pack(side="left", padx=(0,6))
        self.entry = ttk.Entry(row0, textvariable=self.video_var, state="readonly", width=100)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0,6))
        ttk.Button(row0, text="Open Video…", command=self.choose_video).pack(side="left")

        row1 = ttk.Frame(top); row1.pack(fill="x")
        self.btn_run = ttk.Button(row1, text="Run Classification ▶", command=self.run_threaded)
        self.btn_run.pack(side="left", padx=(0,6))
        ttk.Button(row1, text="Open Output Folder", command=self.open_results).pack(side="left")

        # progress + log
        prog = ttk.Frame(self); prog.pack(fill="x", padx=pad, pady=(0,6))
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=5)
        self.progress.pack(fill="x")
        ttk.Label(prog, textvariable=self.step_var).pack(anchor="w")

        self.log = tk.Text(self, height=8, wrap="word", state="disabled")
        self.log.pack(fill="x", padx=pad, pady=(0,8))

        # tabs
        self.tabs = ttk.Notebook(self); self.tabs.pack(fill="both", expand=True, padx=pad, pady=(0,pad))

        self.tab_pre = ttk.Frame(self.tabs); self.tabs.add(self.tab_pre, text="1) Preprocess")
        ttk.Label(self.tab_pre, text="Preprocessed Frame", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_pre = ttk.Label(self.tab_pre, text="(Run to see a sample preprocessed frame)")
        self.lbl_pre.pack(fill="both", expand=True)

        self.tab_seg = ttk.Frame(self.tabs); self.tabs.add(self.tab_seg, text="2) Segmentation")
        ttk.Label(self.tab_seg, text="Segmentation Overlay", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_seg = ttk.Label(self.tab_seg, text="(Run to see segmentation overlay)")
        self.lbl_seg.pack(fill="both", expand=True)

        self.tab_cls = ttk.Frame(self.tabs); self.tabs.add(self.tab_cls, text="3) Classification")
        ttk.Label(self.tab_cls, text="Prediction", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_pred = ttk.Label(self.tab_cls, textvariable=self.pred_var, font=("Segoe UI", 18, "bold"))
        self.lbl_pred.pack(anchor="w", pady=(2,6))
        ttk.Label(self.tab_cls, text="Votes").pack(anchor="w")
        ttk.Label(self.tab_cls, textvariable=self.vote_var).pack(anchor="w")

        # menu + shortcut
        menubar = tk.Menu(self)
        runmenu = tk.Menu(menubar, tearoff=0)
        runmenu.add_command(label="Run Classification ▶", command=self.run_threaded, accelerator="Ctrl+R")
        menubar.add_cascade(label="Run", menu=runmenu)
        self.config(menu=menubar)
        self.bind_all("<Control-r>", lambda e: self.run_threaded())

    # helpers
    def log_write(self, s):
        self.log.configure(state="normal")
        self.log.insert("end", s + "\n")
        self.log.configure(state="disabled")
        self.log.see("end")

    def choose_video(self):
        p = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")]
        )
        if not p: return
        self.video_path = p
        self.video_var.set(p)

    def open_results(self):
        path = self.results_root
        try:
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception:
            messagebox.showinfo("Results", path)

    def run_threaded(self):
        if not self.video_path:
            messagebox.showwarning("No video", "Please choose a video first.")
            return
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        try:
            self.progress["value"] = 0
            self.step_var.set("Starting…")
            self.log.configure(state="normal"); self.log.delete("1.0", "end"); self.log.configure(state="disabled")

            def tick(msg):
                self.progress["value"] += 1
                self.step_var.set(msg)
                self.log_write(f"[STEP] {msg}")

            tick("Running pipeline")
            out = run_classification_pipeline(self.video_path, self.results_root, log_fn=self.log_write)

            # previews
            frames_dir = out["dirs"]["frames"]
            seg_dir = out["dirs"]["seg"]

            frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
            masks = sorted([f for f in os.listdir(seg_dir) if f.endswith(".png")])
            if frames:
                mid = frames[len(frames)//2]
                img = cv2.imread(os.path.join(frames_dir, mid), cv2.IMREAD_GRAYSCALE)
                self.img_pre = to_tk(img)
                self.lbl_pre.configure(image=self.img_pre, text="")
            if frames and masks:
                i = min(len(frames), len(masks)) // 2
                ov = overlay_edges(os.path.join(frames_dir, frames[i]), os.path.join(seg_dir, masks[i]))
                if ov:
                    self.img_seg = ov
                    self.lbl_seg.configure(image=self.img_seg, text="")

            # results
            classification = out["classification"]
            votes = out["votes"]
            self.pred_var.set(classification)
            self.lbl_pred.configure(foreground=("#c62828" if classification.startswith("Cancer") else "#1565c0"))
            self.vote_var.set(f"Cancer: {votes['Cancer']} | Normal: {votes['Normal']}")
            tick("Done")
            self.tabs.select(self.tab_cls)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log_write(f"[ERROR] {e}")

if __name__ == "__main__":
    App().mainloop()
