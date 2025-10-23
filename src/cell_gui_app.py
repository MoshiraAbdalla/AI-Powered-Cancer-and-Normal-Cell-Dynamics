#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
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


# ────────── helpers for UI images ──────────
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
    """Purely a VIEW helper (does not change your segmentation)."""
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


def draw_tracks_overlay(base_frame_path, linked_df, size=(520, 380)):
    frame = cv2.imread(base_frame_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    try:
        for _, g in linked_df.groupby("particle"):
            g = g.sort_values("frame")
            pts = g[["x", "y"]].values.astype(int)
            for i in range(1, len(pts)):
                cv2.line(img, tuple(pts[i-1]), tuple(pts[i]), (0, 255, 0), 1)
    except Exception:
        pass
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))


# ────────── pipeline (your exact logic; added preview_fn calls only) ──────────
def run_pipeline(video_path, results_root,
                 log_fn=print,
                 progress_fn=lambda *_: None,
                 preview_fn=lambda *_: None):
    """
    Uses only the code you provided: extract frames -> Otsu segmentation -> centroid detection ->
    trackpy link_df (with SubnetOversizeException retry) -> feature means -> voting model.
    Emits live previews via preview_fn({...}) for the GUI.
    """
    run_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(results_root, run_name)
    frames_dir = os.path.join(out_dir, "frames")
    seg_dir = os.path.join(out_dir, "segmentation")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    # 1) Frames
    progress_fn(1, "Extracting frames…")
    log_fn("\n[INFO] Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    res = (int(cap.get(3)), int(cap.get(4)))
    log_fn(f"[INFO] Video Metadata:\nFrames: {frame_count}, FPS: {fps}, Resolution: {res}")

    i = 0
    PREV_EVERY = 10  # preview every N frames to keep UI responsive
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fpath = os.path.join(frames_dir, f"frame_{i:03d}.png")
        io.imsave(fpath, gray)

        if i % PREV_EVERY == 0:
            # live preview of preprocessed frame
            preview_fn({"type": "pre_frame", "path": fpath})
        if frame_count:
            frac = min(0.99, (i + 1) / frame_count)
            preview_fn({"type": "progress", "base": 0, "frac": frac, "msg": f"Extracting frames… ({i+1}/{frame_count})"})

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
            preview_fn({
                "type": "seg_overlay",
                "frame_path": os.path.join(frames_dir, f),
                "mask_path": mpath
            })
        total = max(1, len(frame_files))
        frac = (idx + 1) / total
        preview_fn({"type": "progress", "base": 1, "frac": frac, "msg": f"Segmenting cells… ({idx+1}/{total})"})

    log_fn(f"[INFO] Segmentation complete – saved masks to '{seg_dir}'")
    mask_files = sorted([f for f in os.listdir(seg_dir) if f.endswith(".png")])

    # 3) Tracking
    progress_fn(3, "Tracking cells…")
    log_fn("\n[INFO] Tracking cells...")
    all_data = []
    for idx, f in enumerate(mask_files):
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
        log_fn(f"[WARN] {e}. Retrying with smaller search range…")
        linked = tp.link_df(df, search_range=15, memory=memory)

    cells = int(linked["particle"].nunique()) if "particle" in linked.columns else 0
    log_fn(f"[INFO] Tracking complete – {cells} cells identified.")

    # show a tracking overlay immediately
    if frame_files:
        last_frame_path = os.path.join(frames_dir, frame_files[-1])
        preview_fn({"type": "tracking_overlay", "frame_path": last_frame_path, "linked_df": linked})
    preview_fn({"type": "progress", "base": 2, "frac": 1.0, "msg": "Tracking cells… done"})

    # 4) Features
    preview_fn({"type": "progress", "base": 3, "frac": 0.2, "msg": "Computing features…"})
    log_fn("\n[INFO] Computing motion features...")
    features = []
    for pid, g in linked.groupby("particle"):
        if len(g) < 3:
            continue
        g = g.sort_values("frame")
        dx = np.diff(g["x"]); dy = np.diff(g["y"])
        speed = np.mean(np.sqrt(dx**2 + dy**2))
        disp = np.sqrt((g["x"].iloc[-1] - g["x"].iloc[0])**2 + (g["y"].iloc[-1] - g["y"].iloc[0])**2)
        ang = np.mean(np.abs(np.degrees(np.arctan2(dy, dx))))
        features.append([pid, speed, disp, ang])
    feat_df = pd.DataFrame(features, columns=["particle", "speed", "disp", "angle"])
    preview_fn({"type": "progress", "base": 3, "frac": 1.0, "msg": "Computing features… done"})
    log_fn(f"[INFO] Computed features for {len(feat_df)} cells.")

    if len(feat_df) == 0:
        mean_speed = 0.0; mean_disp = 0.0; mean_angle = 0.0
    else:
        mean_speed = float(feat_df["speed"].mean())
        mean_disp  = float(feat_df["disp"].mean())
        mean_angle = float(feat_df["angle"].mean())

    log_fn(f"\n[INFO] Video-Level Statistics:\nMean Speed: {mean_speed:.2f} px/s")
    log_fn(f"Total Displacement: {mean_disp:.2f} px")
    log_fn(f"Mean Turn Angle: {mean_angle:.2f}°")

    # 5) Classification (voting model)
    preview_fn({"type": "progress", "base": 4, "frac": 0.4, "msg": "Classifying…"})
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
    preview_fn({"type": "progress", "base": 4, "frac": 1.0, "msg": "Classifying… done"})

    log_fn(f"\n[RESULT] Classification: {classification}")
    log_fn(f"[INFO] Cancer-Likeness Score: {cancer_likeness:.2f}")
    log_fn(f"[INFO] Feature Votes: {votes}")

    # preview frame paths
    previews = {
        "frames_dir": frames_dir,
        "seg_dir": seg_dir,
        "frame_files": frame_files,
        "mask_files": mask_files,
        "linked_df": linked
    }
    return classification, votes, (mean_speed, mean_disp, mean_angle), previews


# ────────── GUI ──────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cell Movement Classifier — Stepwise (Single-file)")
        self.geometry("1180x780")
        self.minsize(1120, 740)

        self.results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
        os.makedirs(self.results_root, exist_ok=True)

        self.video_path = None
        self.video_var = tk.StringVar(value="No video selected")
        self.pred_var = tk.StringVar(value="—")
        self.vote_var = tk.StringVar(value="Cancer: 0 | Normal: 0")
        self.step_var = tk.StringVar(value="Ready")

        # recording state
        self.is_rec = False
        self.cap = None
        self.writer = None
        self.record_path = None

        # keep images alive
        self.img_pre = None
        self.img_seg = None
        self.img_track = None

        self.build_ui()

    # ---------- thread-safe preview receiver ----------
    def preview_cb(self, evt: dict):
        """Receive live preview/progress events from the worker thread."""
        def _apply():
            typ = evt.get("type")
            if typ == "pre_frame":
                img = cv2.imread(evt["path"], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.img_pre = to_tk(img)
                    self.lbl_pre.configure(image=self.img_pre, text="")
                    self.tabs.select(self.tab_pre)

            elif typ == "seg_overlay":
                ov = overlay_edges(evt["frame_path"], evt["mask_path"])
                if ov:
                    self.img_seg = ov
                    self.lbl_seg.configure(image=self.img_seg, text="")
                    self.tabs.select(self.tab_seg)

            elif typ == "tracking_overlay":
                trk = draw_tracks_overlay(evt["frame_path"], evt["linked_df"])
                if trk:
                    self.img_track = trk
                    self.lbl_track.configure(image=self.img_track, text="")
                    self.tabs.select(self.tab_track)

            elif typ == "log":
                self.log_write(evt.get("msg", ""))

            elif typ == "progress":
                base = evt.get("base", 0)   # 0..4 (before each step)
                frac = float(evt.get("frac", 0.0))  # 0..1 inside the step
                # progress bar maximum is 5 (five steps). Set value to base+frac.
                self.progress["value"] = min(5, base + frac)
                msg = evt.get("msg")
                if msg:
                    self.step_var.set(msg)
        self.after(0, _apply)

    def build_ui(self):
        pad = 10

        # Top (two rows)
        top = ttk.Frame(self); top.pack(fill="x", padx=pad, pady=(pad, 4))
        row0 = ttk.Frame(top); row0.pack(fill="x", pady=(0,6))
        ttk.Label(row0, text="Selected:").pack(side="left", padx=(0,6))
        self.entry = ttk.Entry(row0, textvariable=self.video_var, state="readonly", width=100)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0,6))
        ttk.Button(row0, text="Open Video…", command=self.choose_video).pack(side="left")

        row1 = ttk.Frame(top); row1.pack(fill="x")
        self.btn_start = ttk.Button(row1, text="Start Live Recording", command=self.start_recording)
        self.btn_stop  = ttk.Button(row1, text="Stop Recording", command=self.stop_recording, state="disabled")
        self.btn_run   = ttk.Button(row1, text="Run Classification ▶", command=self.run_threaded)
        self.btn_open  = ttk.Button(row1, text="Open Output Folder", command=self.open_results)
        self.btn_start.pack(side="left", padx=(0,6))
        self.btn_stop.pack(side="left", padx=(0,6))
        self.btn_run.pack(side="left", padx=(0,6))
        self.btn_open.pack(side="left", padx=(0,6))

        # progress + log (5 steps total)
        prog = ttk.Frame(self); prog.pack(fill="x", padx=pad, pady=(0,6))
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=5, value=0)
        self.progress.pack(fill="x")
        ttk.Label(prog, textvariable=self.step_var).pack(anchor="w")
        self.log = tk.Text(self, height=10, wrap="word", state="disabled")
        self.log.pack(fill="x", padx=pad, pady=(0,8))

        # tabs
        self.tabs = ttk.Notebook(self); self.tabs.pack(fill="both", expand=True, padx=pad, pady=(0,pad))

        self.tab_pre = ttk.Frame(self.tabs); self.tabs.add(self.tab_pre, text="1) Preprocess")
        ttk.Label(self.tab_pre, text="Preprocessed Frame", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_pre = ttk.Label(self.tab_pre, text="(Live preview during recording; sample frame after preprocessing)")
        self.lbl_pre.pack(fill="both", expand=True)

        self.tab_seg = ttk.Frame(self.tabs); self.tabs.add(self.tab_seg, text="2) Segmentation")
        ttk.Label(self.tab_seg, text="Segmentation Overlay", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_seg = ttk.Label(self.tab_seg, text="(Run to see segmentation overlay)")
        self.lbl_seg.pack(fill="both", expand=True)

        self.tab_track = ttk.Frame(self.tabs); self.tabs.add(self.tab_track, text="3) Tracking")
        ttk.Label(self.tab_track, text="Trajectory Overlay", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_track = ttk.Label(self.tab_track, text="(Run to see trajectories)")
        self.lbl_track.pack(fill="both", expand=True)

        self.tab_cls = ttk.Frame(self.tabs); self.tabs.add(self.tab_cls, text="4) Classification")
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

    # ────── utility ──────
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

    # ────── recording ──────
    def start_recording(self):
        if self.is_rec: return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return
        self.cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        rec_dir = os.path.join(self.results_root, "recordings"); os.makedirs(rec_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.record_path = os.path.join(rec_dir, f"record_{ts}.mp4")
        self.writer = cv2.VideoWriter(self.record_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        self.is_rec = True
        self.btn_start.configure(state="disabled"); self.btn_stop.configure(state="normal"); self.btn_run.configure(state="disabled")
        threading.Thread(target=self._record_loop, daemon=True).start()
        self.tabs.select(self.tab_pre)
        self.log_write("[REC] Recording started. Live preview on Preprocess tab.")

    def _record_loop(self):
        while self.is_rec and self.cap and self.writer:
            ok, frame = self.cap.read()
            if not ok: break
            self.writer.write(frame)
            photo = to_tk(frame)
            self.after(0, lambda p=photo: (self.lbl_pre.configure(image=p, text=""), setattr(self, "img_pre", p)))
            time.sleep(0.001)

    def stop_recording(self):
        if not self.is_rec: return
        self.is_rec = False
        try:
            if self.cap: self.cap.release()
            if self.writer: self.writer.release()
        finally:
            self.cap = None; self.writer = None
        self.btn_start.configure(state="normal"); self.btn_stop.configure(state="disabled"); self.btn_run.configure(state="normal")
        if self.record_path and os.path.exists(self.record_path):
            self.video_path = self.record_path
            self.video_var.set(self.record_path)
            self.log_write(f"[REC] Saved {self.record_path}")

    # ────── pipeline run ──────
    def run_threaded(self):
        if self.is_rec:
            messagebox.showinfo("Recording", "Stop recording before running classification.")
            return
        if not self.video_path:
            messagebox.showwarning("No video", "Please choose or record a video first.")
            return
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        try:
            # reset UI
            self.progress.configure(value=0, maximum=5)
            self.step_var.set("Starting…")
            self.log.configure(state="normal"); self.log.delete("1.0", "end"); self.log.configure(state="disabled")
            self.lbl_seg.configure(image="", text="(Run to see segmentation overlay)")
            self.lbl_track.configure(image="", text="(Run to see trajectories)")
            self.pred_var.set("—"); self.vote_var.set("Cancer: 0 | Normal: 0")

            def progress(step, msg):
                self.progress["value"] = step
                self.step_var.set(msg)
                self.log_write(f"[STEP] {msg}")

            classification, votes, means, previews = run_pipeline(
                self.video_path,
                self.results_root,
                log_fn=self.log_write,
                progress_fn=progress,
                preview_fn=self.preview_cb,  # <<< live updates
            )

            # Final classification results
            self.pred_var.set(classification)
            self.lbl_pred.configure(foreground=("#c62828" if classification.startswith("Cancer") else "#1565c0"))
            self.vote_var.set(f"Cancer: {votes['Cancer']} | Normal: {votes['Normal']}")

            # Finish bar
            self.progress["value"] = 5
            self.step_var.set("Done")
            self.tabs.select(self.tab_cls)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log_write(f"[ERROR] {e}")


if __name__ == "__main__":
    App().mainloop()
