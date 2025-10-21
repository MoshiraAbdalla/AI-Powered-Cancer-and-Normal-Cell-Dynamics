#!/usr/bin/env python3
"""
Cell Movement Classifier GUI — Stepwise View (Preprocess → Segmentation → Tracking → Classification)
- Live recording preview (Preprocess tab)
- Distinct previews for each step
- Voting model classification using existing repo logic
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Ensure src/ modules import
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(THIS_DIR, "src"), os.path.join(THIS_DIR, "src", "src")]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Project modules (from your repo)
from preprocess import preprocess_video
from segment import segment_cells
from track import track_cells
from features import compute_motion_features
from metadata import extract_metadata

import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageTk


def to_tk_image(img_bgr_or_gray, size=(520, 380)):
    """Convert BGR or GRAY np array to ImageTk.PhotoImage resized for UI."""
    if img_bgr_or_gray is None:
        return None
    if len(img_bgr_or_gray.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
    if size:
        img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    img = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(img)


def overlay_edges(gray_path, mask_path, out_size=(520, 380)):
    """Create a color overlay image: original gray + red mask edges."""
    try:
        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            return None
        edges = cv2.Canny(mask, 50, 150)
        gray_col = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay = gray_col.copy()
        overlay[edges > 0] = (0, 0, 255)
        blended = cv2.addWeighted(gray_col, 0.85, overlay, 0.15, 0)
        if out_size:
            blended = cv2.resize(blended, out_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception:
        return None


def draw_tracks_on_frame(frame_path, tracks_csv, size=(520, 380)):
    """Overlay simple polylines for each particle track on top of a frame."""
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None
    frame_col = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    try:
        df = pd.read_csv(tracks_csv)
        # Expect columns: frame, x, y, particle (or similar)
        # Normalize column names
        cols = {c.lower(): c for c in df.columns}
        frame_colname = cols.get('frame') or cols.get('frame_index') or 'frame'
        x_col = cols.get('x') or 'x'
        y_col = cols.get('y') or 'y'
        pid_col = cols.get('particle') or cols.get('id') or 'particle'

        for pid, g in df.groupby(pid_col):
            pts = g.sort_values(frame_colname)[[x_col, y_col]].values.astype(int)
            for i in range(1, len(pts)):
                cv2.line(frame_col, tuple(pts[i-1]), tuple(pts[i]), (0, 255, 0), 1)
    except Exception:
        # best effort; if format mismatch just return the raw frame
        pass

    if size:
        frame_col = cv2.resize(frame_col, size, interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame_col, cv2.COLOR_BGR2RGB)))


class MetricCard(ttk.Frame):
    def __init__(self, master, title):
        super().__init__(master, padding=8)
        self.configure(style="Card.TFrame")
        self.value_var = tk.StringVar(value="—")
        ttk.Label(self, text=title, font=("Segoe UI", 10)).pack(anchor="w")
        ttk.Label(self, textvariable=self.value_var, font=("Segoe UI", 16, "bold")).pack(anchor="w")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cell Movement Classifier — Stepwise")
        self.geometry("1180x760")
        self.minsize(1120, 720)

        self.video_path = None
        self.results_root = os.path.join(THIS_DIR, "ResultsGUI")
        os.makedirs(self.results_root, exist_ok=True)

        # UI state
        self.pred_var = tk.StringVar(value="—")
        self.step_var = tk.StringVar(value="Ready")
        self.vote_var = tk.StringVar(value="Cancer: 0 | Normal: 0")
        self.thresh_var = tk.StringVar(value="—")
        self.mean_speed = tk.StringVar(value="—")
        self.mean_disp = tk.StringVar(value="—")
        self.mean_angle = tk.StringVar(value="—")

        # Live recording state
        self.is_recording = False
        self.cap = None
        self.writer = None
        self.record_path = None
        self.stream_thread = None

        # Image refs
        self.img_pre = None
        self.img_seg = None
        self.img_track = None
        self.img_class = None

        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Card.TFrame", background="#ffffff", relief="groove")
        style.configure("Pred.Good.TLabel", foreground="#1565c0", font=("Segoe UI", 18, "bold"))
        style.configure("Pred.Bad.TLabel", foreground="#c62828", font=("Segoe UI", 18, "bold"))

    def _build_ui(self):
        pad = 10
        # Top controls
        top = ttk.Frame(self)
        top.pack(fill="x", padx=pad, pady=(pad, 4))

        self.video_lbl = ttk.Label(top, text="No video selected")
        self.video_lbl.pack(side="left", padx=(0, 8))

        ttk.Button(top, text="Open Video…", command=self.choose_video).pack(side="left", padx=(0, 6))
        self.btn_start = ttk.Button(top, text="Start Live Recording", command=self.start_recording)
        self.btn_stop = ttk.Button(top, text="Stop Recording", command=self.stop_recording, state="disabled")
        self.btn_start.pack(side="left", padx=(0, 6))
        self.btn_stop.pack(side="left", padx=(0, 6))
        self.btn_run = ttk.Button(top, text="Run Classification", command=self.run_pipeline_threaded)
        self.btn_run.pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Open Output Folder", command=self.open_output_folder).pack(side="left")

        # Progress + log
        prog = ttk.Frame(self)
        prog.pack(fill="x", padx=pad, pady=(0, 6))
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=5)
        self.progress.pack(fill="x")
        ttk.Label(prog, textvariable=self.step_var).pack(anchor="w")

        self.log = tk.Text(self, height=8, wrap="word", state="disabled")
        self.log.pack(fill="x", padx=pad, pady=(0,8))

        # Tabs for each step
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=pad, pady=(0, pad))

        # Preprocess tab
        self.tab_pre = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_pre, text="1) Preprocess")
        ttk.Label(self.tab_pre, text="Preprocessed Frame", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_pre = ttk.Label(self.tab_pre, text="(Live stream shows here while recording. After preprocessing, a sample frame appears.)")
        self.lbl_pre.pack(fill="both", expand=True)

        # Segmentation tab
        self.tab_seg = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_seg, text="2) Segmentation")
        ttk.Label(self.tab_seg, text="Segmentation Overlay", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_seg = ttk.Label(self.tab_seg, text="(Run classification to generate segmentation preview)")
        self.lbl_seg.pack(fill="both", expand=True)

        # Tracking tab
        self.tab_track = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_track, text="3) Tracking")
        ttk.Label(self.tab_track, text="Trajectory Overlay", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_track = ttk.Label(self.tab_track, text="(Run classification to generate tracking preview)")
        self.lbl_track.pack(fill="both", expand=True)

        # Classification tab
        self.tab_class = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_class, text="4) Classification")
        top_class = ttk.Frame(self.tab_class)
        top_class.pack(fill="x", pady=(8,6))
        ttk.Label(top_class, text="Prediction", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.pred_label = ttk.Label(top_class, textvariable=self.pred_var, style="Pred.Good.TLabel")
        self.pred_label.pack(anchor="w", pady=(2, 6))

        vb = ttk.Frame(self.tab_class)
        vb.pack(fill="x")
        ttk.Label(vb, text="Votes", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lbl_votes = ttk.Label(vb, textvariable=self.vote_var)
        self.lbl_votes.pack(anchor="w", pady=(2,0))

        th = ttk.Frame(self.tab_class)
        th.pack(fill="x", pady=(4, 8))
        ttk.Label(th, text="Thresholds (midpoints)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lbl_thresh = ttk.Label(th, textvariable=self.thresh_var, wraplength=520, justify="left")
        self.lbl_thresh.pack(anchor="w")

        cards = ttk.Frame(self.tab_class)
        cards.pack(fill="x")
        self.card_speed = MetricCard(cards, "Mean Speed (px/s)")
        self.card_disp = MetricCard(cards, "Mean Displacement (px)")
        self.card_angle = MetricCard(cards, "Mean Turn Angle (°)")
        self.card_speed.pack(fill="x", pady=4)
        self.card_disp.pack(fill="x", pady=4)
        self.card_angle.pack(fill="x", pady=4)

    # ----- helpers -----
    def set_pred_style(self, pred):
        if pred.lower().startswith("cancer"):
            self.pred_label.configure(style="Pred.Bad.TLabel")
        else:
            self.pred_label.configure(style="Pred.Good.TLabel")

    def _log_reset(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.configure(state="disabled")
        self.log.see("end")

    def _tick(self, msg):
        self.progress["value"] += 1
        self.step_var.set(msg)
        self._log(f"[STEP] {msg}")

    def _reset_progress(self):
        self.progress["value"] = 0
        self.step_var.set("Starting…")
        self._log_reset()

    # ----- file actions -----
    def choose_video(self):
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.video_path = path
        self.video_lbl.configure(text=f"Selected: {path}")

    def open_output_folder(self):
        path = self.results_root
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception:
            messagebox.showinfo("Output Folder", f"Results are saved under:\n{path}")

    # ----- live recording -----
    def start_recording(self):
        if self.is_recording:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 20.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        out_dir = os.path.join(self.results_root, "recordings")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.record_path = os.path.join(out_dir, f"record_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.record_path, fourcc, fps, (width, height))

        self.is_recording = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_run.configure(state="disabled")

        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        self._log("[REC] Recording started. Live preview on Preprocess tab.")

        # switch to preprocess tab to show live view
        self.tabs.select(self.tab_pre)

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)

        try:
            if self.cap: self.cap.release()
        except Exception: pass
        try:
            if self.writer: self.writer.release()
        except Exception: pass

        self.cap = None
        self.writer = None

        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_run.configure(state="normal")

        if self.record_path and os.path.exists(self.record_path):
            self.video_path = self.record_path
            self.video_lbl.configure(text=f"Recorded: {self.record_path}")
            self._log(f"[REC] Saved recording to {self.record_path}")
        else:
            self._log("[REC] Recording stopped (no file saved)")

    def _stream_loop(self):
        while self.is_recording and self.cap and self.writer:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.writer.write(frame)

            # Show in preprocess tab
            photo = to_tk_image(frame, size=(520, 380))
            def update_preview(p=photo):
                self.img_pre = p
                self.lbl_pre.configure(image=self.img_pre, text="")
            self.after(0, update_preview)

            time.sleep(0.001)

    # ----- pipeline -----
    def run_pipeline_threaded(self):
        if not self.video_path:
            messagebox.showwarning("No video", "Please choose or record a video first.")
            return
        if self.is_recording:
            messagebox.showinfo("Recording in progress", "Stop recording before running classification.")
            return
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        try:
            self._reset_progress()
            video = self.video_path

            run_name = os.path.splitext(os.path.basename(video))[0]
            out_root = os.path.join(self.results_root, run_name)
            frames_dir = os.path.join(out_root, "frames")
            seg_dir = os.path.join(out_root, "segmentation")
            tracking_csv = os.path.join(out_root, "tracking.csv")
            features_csv = os.path.join(out_root, "features.csv")
            os.makedirs(out_root, exist_ok=True)

            # 0) Metadata
            self._tick("Extracting video metadata")
            info = extract_metadata(video)
            if info is None:
                raise RuntimeError("Failed to read video metadata")
            self._log(f"[VIDEO] {info}")

            # 1) Preprocess -> frames
            self._tick("Preprocessing video to frames")
            preprocess_video(video, frames_dir)

            # show a sample preprocessed frame
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith('.png')])
            if frame_files:
                sample_frame_path = os.path.join(frames_dir, frame_files[len(frame_files)//2])
                img = cv2.imread(sample_frame_path, cv2.IMREAD_GRAYSCALE)
                photo = to_tk_image(img)
                self.img_pre = photo
                self.lbl_pre.configure(image=self.img_pre, text="")
            self.tabs.select(self.tab_pre)

            # 2) Segmentation
            self._tick("Segmenting cells")
            segment_cells(frames_dir, seg_dir)

            # segmentation overlay preview
            mask_files = sorted([f for f in os.listdir(seg_dir) if f.lower().endswith('.png')])
            if frame_files and mask_files:
                idx = min(len(frame_files), len(mask_files)) // 2
                frame_path = os.path.join(frames_dir, frame_files[idx])
                mask_path = os.path.join(seg_dir, mask_files[idx])
                img = overlay_edges(frame_path, mask_path)
                if img:
                    self.img_seg = ImageTk.PhotoImage(img)
                    self.lbl_seg.configure(image=self.img_seg, text="")
            self.tabs.select(self.tab_seg)

            # 3) Tracking
            self._tick("Tracking cells")
            track_cells(seg_dir, tracking_csv)

            # tracking polyline preview
            if frame_files:
                frame_for_tracks = os.path.join(frames_dir, frame_files[-1])
                photo = draw_tracks_on_frame(frame_for_tracks, tracking_csv)
                if photo:
                    self.img_track = photo
                    self.lbl_track.configure(image=self.img_track, text="")
            self.tabs.select(self.tab_track)

            # 4) Features + Classification (Voting)
            self._tick("Computing features & classifying")
            compute_motion_features(tracking_csv, features_csv)

            df = pd.read_csv(features_csv)
            mean_speed_val = float(df['mean_speed(px/s)'].mean())
            mean_disp_val  = float(df['total_displacement(px)'].mean())
            mean_angle_val = float(df['mean_turn_angle(deg)'].mean())

            norm = {'speed': 35.96, 'disp': 26.08, 'angle': 69.7}
            canc = {'speed': 61.95, 'disp': 15.48, 'angle': 90.89}

            th_speed = (norm['speed'] + canc['speed']) / 2
            th_disp  = (norm['disp']  + canc['disp'])  / 2
            th_angle = (norm['angle'] + canc['angle']) / 2

            votes = {"Cancer": 0, "Normal": 0}

            if mean_speed_val > th_speed:
                votes["Cancer"] += 1
            else:
                votes["Normal"] += 1

            if mean_disp_val > th_disp:
                votes["Normal"] += 1
            else:
                votes["Cancer"] += 1

            if mean_angle_val > th_angle:
                votes["Cancer"] += 1
            else:
                votes["Normal"] += 1

            classification = "Cancer Cell" if votes["Cancer"] > votes["Normal"] else "Normal Cell"

            # Update classification tab
            self.pred_var.set(classification)
            self.set_pred_style(classification)
            self.vote_var.set(f"Cancer: {votes['Cancer']} | Normal: {votes['Normal']}")
            self.thresh_var.set(f"speed th={th_speed:.2f}, disp th={th_disp:.2f}, angle th={th_angle:.2f}")
            self.card_speed.value_var.set(f"{mean_speed_val:.2f}")
            self.card_disp.value_var.set(f"{mean_disp_val:.2f}")
            self.card_angle.value_var.set(f"{mean_angle_val:.2f}")

            self._log("[RESULT] " + classification)
            self._log(f"[VOTES] {votes}")
            self._log(f"[STATS] speed={mean_speed_val:.2f}, disp={mean_disp_val:.2f}, angle={mean_angle_val:.2f}")
            self._log(f"[OUTPUT] {out_root}")

            self.tabs.select(self.tab_class)
            self._tick("Done")

        except Exception as e:
            self.pred_var.set("Error")
            self.set_pred_style("Normal")
            messagebox.showerror("Pipeline error", str(e))
            self._log(f"[ERROR] {e}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
