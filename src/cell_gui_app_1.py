#!/usr/bin/env python3
"""
Cell Movement Classifier — GUI (Stepwise)
Preprocess → Segmentation → Tracking → Classification

Design goals
- Reuse your existing pipeline modules in src/: preprocess, segment, track, features, metadata
- Show each step in its own tab with visual previews
- Keep your voting thresholds exactly as written in your code
- Make the top bar two rows so "Run Classification" is always visible
- Add Ctrl+R and a "Run" menu to start the pipeline from keyboard/menu
- Handle trackpy "Subnetworks contains XX points" by falling back automatically (no blocking popup)
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# ── Make sure we can import from ./src ─────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_CANDIDATES = [os.path.join(THIS_DIR, "src"), os.path.join(THIS_DIR, "src", "src")]
for p in SRC_CANDIDATES:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# ── Import your pipeline functions (AS-IS) ─────────────────────────────────────
from preprocess import preprocess_video
from segment import segment_cells
from track import track_cells as repo_track_cells    # we keep your tracker as the first choice
from features import compute_motion_features
from metadata import extract_metadata

# ── Third-party libs used only for UI/preview + robust fallback ────────────────
import cv2
import pandas as pd
from PIL import Image, ImageTk

# Fallback tracking uses trackpy, but only if your tracker fails
import trackpy as tp
try:
    from trackpy.linking.utils import SubnetOversizeException
except Exception:
    class SubnetOversizeException(Exception):
        pass


# ╔═══════════════════════════════
# ║ Small helpers for preview only
# ╚═══════════════════════════════
def _to_tk_image(img_bgr_or_gray, size=(520, 380)):
    """Convert BGR or GRAY ndarray to ImageTk.PhotoImage resized for UI."""
    if img_bgr_or_gray is None:
        return None
    if len(img_bgr_or_gray.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
    if size:
        img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(img_rgb))


def _overlay_edges(gray_path, mask_path, out_size=(520, 380)):
    """Create a color overlay image: original gray + red mask edges (for display only)."""
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
        return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)))
    except Exception:
        return None


def _draw_tracks_on_frame(frame_path, tracks_csv, size=(520, 380)):
    """Draw simple polylines for each particle trajectory (for display only)."""
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    try:
        df = pd.read_csv(tracks_csv)
        cols = {c.lower(): c for c in df.columns}
        col_f = cols.get('frame') or cols.get('frame_index') or 'frame'
        col_x = cols.get('x') or 'x'
        col_y = cols.get('y') or 'y'
        col_id = cols.get('particle') or cols.get('id') or 'particle'
        for _, g in df.groupby(col_id):
            g = g.sort_values(col_f)
            pts = g[[col_x, col_y]].values
            for i in range(1, len(pts)):
                p0 = tuple(map(int, pts[i-1]))
                p1 = tuple(map(int, pts[i]))
                cv2.line(img, p0, p1, (0, 255, 0), 1)
    except Exception:
        pass

    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))


# ╔══════════════════════════════════════════════════════════════════════════════
# ║ Fallback tracking — only used if your repo tracker raises an exception
# ╚══════════════════════════════════════════════════════════════════════════════
def _robust_tracking_from_masks(segmentation_dir, output_csv,
                                min_area=100, search_range=12, memory=3, max_subnetwork_size=500):
    """
    Build detections from binary masks (connected components), then link with trackpy.
    This avoids SubnetOversizeException by allowing larger subnetworks and shrinking search range a bit.
    """
    frame_files = sorted(f for f in os.listdir(segmentation_dir) if f.lower().endswith('.png'))
    rows = []
    for i, fname in enumerate(frame_files):
        path = os.path.join(segmentation_dir, fname)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        _, binm = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binm, connectivity=8)
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            cx, cy = centroids[lab]
            rows.append({"frame": i, "x": float(cx), "y": float(cy)})

    if not rows:
        raise RuntimeError("Fallback tracker found no centroids. Try reducing min_area.")

    df = pd.DataFrame(rows)
    linked = tp.link_df(
        df,
        search_range=search_range,
        memory=memory,
        neighbor_strategy="KDTree",
        max_subnetwork_size=max_subnetwork_size
    )
    linked.to_csv(output_csv, index=False)
    return linked


# ╔═══════════════════════════════════
# ║ GUI
# ╚═══════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cell Movement Classifier — Stepwise")
        self.geometry("1180x760")
        self.minsize(1120, 720)

        # State
        self.video_path = None
        self.results_root = os.path.join(THIS_DIR, "ResultsGUI")
        os.makedirs(self.results_root, exist_ok=True)

        self.pred_var = tk.StringVar(value="—")
        self.step_var = tk.StringVar(value="Ready")
        self.vote_var = tk.StringVar(value="Cancer: 0 | Normal: 0")
        self.thresh_var = tk.StringVar(value="—")
        self.video_var = tk.StringVar(value="No video selected")   # shown in read-only entry

        # Live recording
        self.is_recording = False
        self.cap = None
        self.writer = None
        self.record_path = None

        # Img refs (to prevent GC)
        self.img_pre = None
        self.img_seg = None
        self.img_track = None

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

        # Top section with two rows so buttons never get pushed off-screen
        top = ttk.Frame(self); top.pack(fill="x", padx=pad, pady=(pad, 4))

        # Row 0 — path + open button
        row0 = ttk.Frame(top); row0.pack(fill="x", pady=(0, 6))
        ttk.Label(row0, text="Selected:").pack(side="left", padx=(0, 6))
        self.path_entry = ttk.Entry(row0, textvariable=self.video_var, state="readonly", width=100)
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(row0, text="Open Video…", command=self.choose_video).pack(side="left")

        # Row 1 — main controls
        row1 = ttk.Frame(top); row1.pack(fill="x")
        self.btn_start = ttk.Button(row1, text="Start Live Recording", command=self.start_recording)
        self.btn_stop  = ttk.Button(row1, text="Stop Recording", command=self.stop_recording, state="disabled")
        self.btn_run   = ttk.Button(row1, text="Run Classification ▶", command=self.run_pipeline_threaded)
        self.btn_open  = ttk.Button(row1, text="Open Output Folder", command=self.open_output_folder)
        self.btn_start.pack(side="left", padx=(0,6))
        self.btn_stop.pack(side="left", padx=(0,6))
        self.btn_run.pack(side="left", padx=(0,6))
        self.btn_open.pack(side="left", padx=(0,6))

        # Progress + log
        prog = ttk.Frame(self); prog.pack(fill="x", padx=pad, pady=(0, 6))
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=5); self.progress.pack(fill="x")
        ttk.Label(prog, textvariable=self.step_var).pack(anchor="w")
        self.log = tk.Text(self, height=8, wrap="word", state="disabled"); self.log.pack(fill="x", padx=pad, pady=(0,8))

        # Tabs
        self.tabs = ttk.Notebook(self); self.tabs.pack(fill="both", expand=True, padx=pad, pady=(0,pad))

        # 1) Preprocess tab
        self.tab_pre = ttk.Frame(self.tabs); self.tabs.add(self.tab_pre, text="1) Preprocess")
        ttk.Label(self.tab_pre, text="Preprocessed Frame", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_pre = ttk.Label(self.tab_pre, text="(Live preview during recording; sample frame after preprocessing)")
        self.lbl_pre.pack(fill="both", expand=True)

        # 2) Segmentation tab
        self.tab_seg = ttk.Frame(self.tabs); self.tabs.add(self.tab_seg, text="2) Segmentation")
        ttk.Label(self.tab_seg, text="Segmentation Overlay", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_seg = ttk.Label(self.tab_seg, text="(Run classification to generate segmentation preview)")
        self.lbl_seg.pack(fill="both", expand=True)

        # 3) Tracking tab
        self.tab_track = ttk.Frame(self.tabs); self.tabs.add(self.tab_track, text="3) Tracking")
        ttk.Label(self.tab_track, text="Trajectory Overlay", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(8,4))
        self.lbl_track = ttk.Label(self.tab_track, text="(Run classification to generate tracking preview)")
        self.lbl_track.pack(fill="both", expand=True)

        # 4) Classification tab
        self.tab_class = ttk.Frame(self.tabs); self.tabs.add(self.tab_class, text="4) Classification")
        top_class = ttk.Frame(self.tab_class); top_class.pack(fill="x", pady=(8,6))
        ttk.Label(top_class, text="Prediction", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.pred_label = ttk.Label(top_class, textvariable=self.pred_var, style="Pred.Good.TLabel")
        self.pred_label.pack(anchor="w", pady=(2,6))

        vb = ttk.Frame(self.tab_class); vb.pack(fill="x")
        ttk.Label(vb, text="Votes", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lbl_votes = ttk.Label(vb, textvariable=self.vote_var); self.lbl_votes.pack(anchor="w", pady=(2,0))

        th = ttk.Frame(self.tab_class); th.pack(fill="x", pady=(4,8))
        ttk.Label(th, text="Thresholds (midpoints)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lbl_thresh = ttk.Label(th, textvariable=self.thresh_var, wraplength=520, justify="left")
        self.lbl_thresh.pack(anchor="w")

        # Metric cards (values shown in logs and here if you later expose per-cell stats)
        cards = ttk.Frame(self.tab_class); cards.pack(fill="x")
        self.card_speed = self._metric(cards, "Mean Speed (px/s)")
        self.card_disp  = self._metric(cards, "Mean Displacement (px)")
        self.card_angle = self._metric(cards, "Mean Turn Angle (°)")

        # Menu + shortcut
        menubar = tk.Menu(self)
        runmenu = tk.Menu(menubar, tearoff=0)
        runmenu.add_command(label="Run Classification ▶", command=self.run_pipeline_threaded, accelerator="Ctrl+R")
        menubar.add_cascade(label="Run", menu=runmenu)
        self.config(menu=menubar)
        self.bind_all("<Control-r>", lambda e: self.run_pipeline_threaded())

    def _metric(self, parent, title):
        f = ttk.Frame(parent, padding=8, style="Card.TFrame"); f.pack(fill="x", pady=4)
        ttk.Label(f, text=title, font=("Segoe UI", 10)).pack(anchor="w")
        v = tk.StringVar(value="—")
        ttk.Label(f, textvariable=v, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        f.value_var = v
        return f

    # ── log & progress helpers ────────────────────────────────────────────────
    def _log_reset(self):
        self.log.configure(state="normal"); self.log.delete("1.0", "end"); self.log.configure(state="disabled")

    def _log(self, text):
        self.log.configure(state="normal"); self.log.insert("end", text + "\n"); self.log.configure(state="disabled"); self.log.see("end")

    def _tick(self, msg):
        self.progress["value"] += 1; self.step_var.set(msg); self._log(f"[STEP] {msg}")

    def _reset_progress(self):
        self.progress["value"] = 0; self.step_var.set("Starting…"); self._log_reset()

    # ── top bar actions ───────────────────────────────────────────────────────
    def choose_video(self):
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.video_path = path
        self.video_var.set(path)
        # Reset previews when switching videos
        self.lbl_pre.configure(image="", text="(Live preview during recording; sample frame after preprocessing)")
        self.lbl_seg.configure(image="", text="(Run classification to generate segmentation preview)")
        self.lbl_track.configure(image="", text="(Run classification to generate tracking preview)")
        self.pred_var.set("—"); self.vote_var.set("Cancer: 0 | Normal: 0"); self.thresh_var.set("—")

    def open_output_folder(self):
        path = self.results_root
        try:
            if sys.platform.startswith("win"): os.startfile(path)
            elif sys.platform == "darwin": os.system(f'open "{path}"')
            else: os.system(f'xdg-open "{path}"')
        except Exception:
            messagebox.showinfo("Output Folder", f"Results are saved under:\n{path}")

    # ── live recording ────────────────────────────────────────────────────────
    def start_recording(self):
        if self.is_recording:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        out_dir = os.path.join(self.results_root, "recordings")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.record_path = os.path.join(out_dir, f"record_{ts}.mp4")
        self.writer = cv2.VideoWriter(self.record_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        self.is_recording = True
        self.btn_start.configure(state="disabled"); self.btn_stop.configure(state="normal"); self.btn_run.configure(state="disabled")
        threading.Thread(target=self._stream_loop, daemon=True).start()
        self._log("[REC] Recording started. Live preview on Preprocess tab.")
        self.tabs.select(self.tab_pre)

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        try:
            if self.cap: self.cap.release()
            if self.writer: self.writer.release()
        finally:
            self.cap = None; self.writer = None
        self.btn_start.configure(state="normal"); self.btn_stop.configure(state="disabled"); self.btn_run.configure(state="normal")
        if self.record_path and os.path.exists(self.record_path):
            self.video_path = self.record_path
            self.video_var.set(self.record_path)
            self._log(f"[REC] Saved {self.record_path}")

    def _stream_loop(self):
        while self.is_recording and self.cap and self.writer:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.writer.write(frame)
            photo = _to_tk_image(frame)
            self.after(0, lambda p=photo: (self.lbl_pre.configure(image=p, text=""), setattr(self, "img_pre", p)))
            time.sleep(0.001)

    # ── pipeline ──────────────────────────────────────────────────────────────
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

            # paths for this run
            run_name = os.path.splitext(os.path.basename(video))[0]
            out_root = os.path.join(self.results_root, run_name)
            frames_dir = os.path.join(out_root, "frames")
            seg_dir = os.path.join(out_root, "segmentation")
            tracks_csv = os.path.join(out_root, "tracking.csv")
            feats_csv = os.path.join(out_root, "features.csv")
            os.makedirs(out_root, exist_ok=True)

            # 0) metadata
            self._tick("Extracting video metadata")
            info = extract_metadata(video)
            if info is None:
                raise RuntimeError("Failed to read video metadata")
            self._log(f"[VIDEO] {info}")

            # 1) preprocess
            self._tick("Preprocessing video to frames")
            preprocess_video(video, frames_dir)

            frame_files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(".png"))
            if frame_files:
                sample = os.path.join(frames_dir, frame_files[len(frame_files)//2])
                pre_img = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
                p = _to_tk_image(pre_img)
                self.img_pre = p; self.lbl_pre.configure(image=self.img_pre, text="")
            self.tabs.select(self.tab_pre)

            # 2) segmentation
            self._tick("Segmenting cells")
            segment_cells(frames_dir, seg_dir)

            mask_files = sorted(f for f in os.listdir(seg_dir) if f.lower().endswith(".png"))
            if frame_files and mask_files:
                i = min(len(frame_files), len(mask_files)) // 2
                seg_overlay = _overlay_edges(os.path.join(frames_dir, frame_files[i]),
                                             os.path.join(seg_dir, mask_files[i]))
                if seg_overlay:
                    self.img_seg = seg_overlay; self.lbl_seg.configure(image=self.img_seg, text="")
            self.tabs.select(self.tab_seg)

            # 3) tracking (try your tracker first, then robust fallback)
            self._tick("Tracking cells")
            tracked_ok = False
            try:
                repo_track_cells(seg_dir, tracks_csv)   # your src tracker
                tracked_ok = True
            except SubnetOversizeException as e:
                self._log(f"[WARN] Trackpy subnetwork too large: {e}. Using fallback.")
            except Exception as e:
                self._log(f"[WARN] Repo tracker failed: {e}. Using fallback.")

            if not tracked_ok:
                _robust_tracking_from_masks(seg_dir, tracks_csv, min_area=120, search_range=12, memory=3, max_subnetwork_size=600)
                self._log("[INFO] Tracking recovered with robust fallback.")

            if frame_files:
                last_frame = os.path.join(frames_dir, frame_files[-1])
                track_img = _draw_tracks_on_frame(last_frame, tracks_csv)
                if track_img:
                    self.img_track = track_img; self.lbl_track.configure(image=self.img_track, text="")
            self.tabs.select(self.tab_track)

            # 4) features + classification (voting model exactly as provided)
            self._tick("Computing features & classifying")
            compute_motion_features(tracks_csv, feats_csv)

            df = pd.read_csv(feats_csv)
            mean_speed = float(df['mean_speed(px/s)'].mean())
            mean_disp  = float(df['total_displacement(px)'].mean())
            mean_angle = float(df['mean_turn_angle(deg)'].mean())

            # Reference stats (unchanged from your snippet)
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

            # Update UI
            self.pred_var.set(classification)
            self.pred_label.configure(style="Pred.Bad.TLabel" if classification.startswith("Cancer") else "Pred.Good.TLabel")
            self.vote_var.set(f"Cancer: {votes['Cancer']} | Normal: {votes['Normal']}")
            self.thresh_var.set(f"speed th={th_speed:.2f}, disp th={th_disp:.2f}, angle th={th_angle:.2f}")

            # (Optional) show numeric metrics on cards — using overall means
            # You can comment these three lines if you prefer hiding them
            self.card_speed.value_var.set(f"{mean_speed:.2f}")
            self.card_disp.value_var.set(f"{mean_disp:.2f}")
            self.card_angle.value_var.set(f"{mean_angle:.2f}")

            self._log(f"[RESULT] {classification}")
            self._log(f"[VOTES] {votes}")
            self._log(f"[STATS] speed={mean_speed:.2f}, disp={mean_disp:.2f}, angle={mean_angle:.2f}")
            self._log(f"[OUTPUT] {out_root}")

            self.tabs.select(self.tab_class)
            self._tick("Done")

        except Exception as e:
            # keep the app responsive even on errors
            self.pred_var.set("Error")
            self.pred_label.configure(style="Pred.Good.TLabel")
            self._log(f"[ERROR] {e}")
            messagebox.showerror("Pipeline error", str(e))

if __name__ == "__main__":
    app = App()
    app.mainloop()
