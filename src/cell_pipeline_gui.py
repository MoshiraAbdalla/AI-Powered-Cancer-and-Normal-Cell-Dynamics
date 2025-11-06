#!/usr/bin/env python3
"""High-resolution desktop GUI for the cell classification pipeline."""

import os
import queue
import threading
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from classify_new_video import *  # noqa: F401,F403

STAGE_TITLES = {
    "preprocess": "Preprocessing",
    "segmentation": "Segmentation",
    "tracking": "Tracking",
    "classification": "Classification",
}


def format_stat(value: float) -> str:
    """Uniform string formatting for statistics."""
    if value is None:
        return "-"
    try:
        if np.isnan(value):
            return "nan"
    except TypeError:
        pass
    return f"{value:.2f}"


class StageView:
    """Helper widget that renders a stage preview with consistent sizing."""

    def __init__(self, parent: ttk.Frame, title: str, size: tuple[int, int] = (420, 320)) -> None:
        self.width, self.height = size
        self.frame = ttk.LabelFrame(parent, text=title, padding=(12, 10))
        self.label = ttk.Label(self.frame, anchor="center", justify="center")
        self.label.pack(expand=True, fill="both")
        self.photo: ImageTk.PhotoImage | None = None
        self.placeholder = "Awaiting run"
        self.show_placeholder()

    def grid(self, **kwargs: Any) -> None:
        self.frame.grid(**kwargs)

    def show_placeholder(self, text: str | None = None) -> None:
        message = text if text is not None else self.placeholder
        self.label.configure(text=message, image="")
        self.photo = None

    def update_image(self, array: np.ndarray | None) -> None:
        if array is None:
            self.show_placeholder("No preview available")
            return
        self.photo = self._array_to_photo(array)
        self.label.configure(image=self.photo, text="")

    def _array_to_photo(self, array: np.ndarray) -> ImageTk.PhotoImage:
        data = array
        if data.ndim == 2:
            if data.dtype == np.bool_:
                data = data.astype(np.uint8) * 255
            else:
                data = data.astype(np.float32)
                max_val = float(data.max()) if data.size else 0.0
                if max_val > 0:
                    data = (data / max_val) * 255.0
                data = np.clip(data, 0, 255).astype(np.uint8)
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        else:
            if data.dtype != np.uint8:
                data = np.clip(data, 0, 255).astype(np.uint8)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(data)
        image = image.resize((self.width, self.height), Image.LANCZOS)
        return ImageTk.PhotoImage(image)


class CellClassifierApp:
    """Tkinter-based GUI that orchestrates the classification pipeline."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AI-Powered Cell Dynamics Explorer")
        self.root.geometry("1400x900")
        self.root.minsize(1220, 820)

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=8)

        self.running = False
        self.ui_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        self.video_path_var = tk.StringVar()
        self.selected_source_var = tk.StringVar(value="No source selected.")
        self.camera_index_var = tk.IntVar(value=0)
        self.max_frames_var = tk.IntVar(value=240)
        self.save_var = tk.BooleanVar(value=False)
        self.frames_dir_var = tk.StringVar(value=os.path.normpath("./Results/gui_frames"))
        self.seg_dir_var = tk.StringVar(value=os.path.normpath("./Results/gui_masks"))

        self.metadata_var = tk.StringVar(value="No video processed yet.")
        self.classification_var = tk.StringVar(value="No classification yet.")
        self.votes_var = tk.StringVar(value="Votes: -")
        self.statistics_summary_var = tk.StringVar(value="Speed / displacement / angle will appear here.")
        self.tracked_cells_var = tk.StringVar(value="Tracked cells: 0")

        self.stage_vars = {
            key: tk.StringVar(value=f"{label}: Waiting")
            for key, label in STAGE_TITLES.items()
        }

        self._build_ui()
        self._toggle_save_options()
        self._append_log("Ready. Load a video or choose a camera to begin.")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._process_queue)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        control = ttk.Frame(self.root, padding=(20, 24))
        control.grid(row=0, column=0, sticky="ns")
        control.columnconfigure(0, weight=1)

        row = 0
        ttk.Label(control, text="Pipeline Source", font=("Segoe UI Semibold", 13)).grid(row=row, column=0, sticky="w")
        row += 1

        self.load_button = ttk.Button(control, text="Load Video...", command=self._choose_video)
        self.load_button.grid(row=row, column=0, sticky="ew", pady=(8, 4))
        row += 1

        ttk.Label(control, textvariable=self.selected_source_var, wraplength=260, justify="left").grid(
            row=row, column=0, sticky="w")
        row += 1

        self.analyze_video_button = ttk.Button(
            control,
            text="Analyze Selected Video",
            command=self._analyze_selected_video,
            style="Accent.TButton",
        )
        self.analyze_video_button.grid(row=row, column=0, sticky="ew", pady=(12, 6))
        row += 1

        ttk.Separator(control).grid(row=row, column=0, sticky="ew", pady=12)
        row += 1

        ttk.Label(control, text="Live Camera", font=("Segoe UI Semibold", 13)).grid(row=row, column=0, sticky="w")
        row += 1

        cam_row = ttk.Frame(control)
        cam_row.grid(row=row, column=0, sticky="ew", pady=(6, 2))
        ttk.Label(cam_row, text="Camera index:").pack(side="left")
        ttk.Spinbox(cam_row, from_=0, to=10, textvariable=self.camera_index_var, width=5, justify="center").pack(
            side="left", padx=(6, 0))
        row += 1

        self.analyze_live_button = ttk.Button(
            control,
            text="Analyze Live Camera",
            command=self._analyze_live_camera,
        )
        self.analyze_live_button.grid(row=row, column=0, sticky="ew", pady=(4, 8))
        row += 1

        ttk.Separator(control).grid(row=row, column=0, sticky="ew", pady=12)
        row += 1

        ttk.Label(control, text="Pipeline Options", font=("Segoe UI Semibold", 13)).grid(row=row, column=0, sticky="w")
        row += 1

        options_frame = ttk.Frame(control)
        options_frame.grid(row=row, column=0, sticky="ew", pady=(6, 2))
        ttk.Label(options_frame, text="Max frames:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(options_frame, from_=0, to=2000, textvariable=self.max_frames_var, width=6, justify="center").grid(
            row=0, column=1, sticky="ew", padx=(6, 0))
        options_frame.columnconfigure(1, weight=1)
        row += 1

        ttk.Checkbutton(control, text="Save intermediate outputs", variable=self.save_var, command=self._toggle_save_options).grid(
            row=row, column=0, sticky="w", pady=(4, 6))
        row += 1

        ttk.Label(control, text="Frames directory:").grid(row=row, column=0, sticky="w")
        row += 1
        self.frames_dir_entry = ttk.Entry(control, textvariable=self.frames_dir_var, width=32)
        self.frames_dir_entry.grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Label(control, text="Masks directory:").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        self.seg_dir_entry = ttk.Entry(control, textvariable=self.seg_dir_var, width=32)
        self.seg_dir_entry.grid(row=row, column=0, sticky="ew")
        row += 1

        ttk.Separator(control).grid(row=row, column=0, sticky="ew", pady=12)
        row += 1

        ttk.Label(control, text="Stage Status", font=("Segoe UI Semibold", 13)).grid(row=row, column=0, sticky="w")
        row += 1

        for key in ("preprocess", "segmentation", "tracking", "classification"):
            ttk.Label(control, textvariable=self.stage_vars[key], wraplength=260, justify="left").grid(
                row=row, column=0, sticky="w", pady=2)
            row += 1

        ttk.Separator(control).grid(row=row, column=0, sticky="ew", pady=12)
        row += 1

        self.progress = ttk.Progressbar(control, mode="indeterminate")
        self.progress.grid(row=row, column=0, sticky="ew")
        row += 1

        self.clear_button = ttk.Button(control, text="Clear Dashboard", command=self._clear_results)
        self.clear_button.grid(row=row, column=0, sticky="ew", pady=(12, 0))

        # ------------------------------------------------------------------
        # Main content area
        # ------------------------------------------------------------------
        content = ttk.Frame(self.root, padding=(20, 24))
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=3)
        content.rowconfigure(1, weight=2)

        stage_container = ttk.Frame(content)
        stage_container.grid(row=0, column=0, sticky="nsew")
        for col in range(3):
            stage_container.columnconfigure(col, weight=1)
        stage_container.rowconfigure(0, weight=1)

        self.stage_views = {
            "preprocess": StageView(stage_container, "1. Preprocessing"),
            "segmentation": StageView(stage_container, "2. Segmentation"),
            "tracking": StageView(stage_container, "3. Tracking"),
        }
        self.stage_views["preprocess"].grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.stage_views["segmentation"].grid(row=0, column=1, sticky="nsew", padx=6)
        self.stage_views["tracking"].grid(row=0, column=2, sticky="nsew", padx=(12, 0))

        bottom = ttk.Frame(content)
        bottom.grid(row=1, column=0, sticky="nsew", pady=(18, 0))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, weight=1)

        info_frame = ttk.Frame(bottom)
        info_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        info_frame.columnconfigure(0, weight=1)

        classification_frame = ttk.LabelFrame(info_frame, text="Classification Summary", padding=(12, 10))
        classification_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(classification_frame, textvariable=self.classification_var, font=("Segoe UI", 20, "bold"), anchor="w").pack(anchor="w")
        ttk.Label(classification_frame, textvariable=self.votes_var, anchor="w").pack(anchor="w", pady=(6, 0))
        ttk.Label(classification_frame, textvariable=self.statistics_summary_var, justify="left", anchor="w").pack(anchor="w", pady=(6, 0))

        metadata_frame = ttk.LabelFrame(info_frame, text="Video Metadata", padding=(12, 10))
        metadata_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(metadata_frame, textvariable=self.metadata_var, justify="left", anchor="w", wraplength=400).pack(anchor="w")

        table_area = ttk.Frame(bottom)
        table_area.grid(row=0, column=1, sticky="nsew")
        table_area.columnconfigure(0, weight=1)
        table_area.rowconfigure(0, weight=1)
        table_area.rowconfigure(1, weight=1)

        features_frame = ttk.LabelFrame(table_area, text="Tracked Cell Features", padding=(12, 10))
        features_frame.grid(row=0, column=0, sticky="nsew")
        features_frame.columnconfigure(0, weight=1)
        ttk.Label(features_frame, textvariable=self.tracked_cells_var, anchor="w").grid(row=0, column=0, sticky="w")

        columns = ("particle", "speed", "disp", "angle")
        self.features_tree = ttk.Treeview(features_frame, columns=columns, show="headings", height=6)
        headings = {
            "particle": "Cell ID",
            "speed": "Mean speed (px/frame)",
            "disp": "Total displacement (px)",
            "angle": "Mean turn angle (deg)",
        }
        widths = {
            "particle": 80,
            "speed": 150,
            "disp": 170,
            "angle": 160,
        }
        for col in columns:
            self.features_tree.heading(col, text=headings[col])
            self.features_tree.column(col, width=widths[col], anchor="center")
        self.features_tree.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        scrollbar = ttk.Scrollbar(features_frame, orient="vertical", command=self.features_tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.features_tree.configure(yscrollcommand=scrollbar.set)

        log_frame = ttk.LabelFrame(table_area, text="Pipeline Log", padding=(12, 10))
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select microscopy video",
            filetypes=[
                ("Video files", "*.avi;*.mp4;*.mov;*.mkv;*.mpg;*.mpeg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.video_path_var.set(path)
        self.selected_source_var.set(f"File: {os.path.basename(path)}")
        self._append_log(f"Selected video file: {path}")

    def _analyze_selected_video(self) -> None:
        path = self.video_path_var.get()
        if not path:
            messagebox.showwarning("Select video", "Please load a video file first.")
            return
        self._start_pipeline(path)

    def _analyze_live_camera(self) -> None:
        index = self.camera_index_var.get()
        self._start_pipeline(int(index))

    def _start_pipeline(self, source: Any) -> None:
        if self.running:
            messagebox.showinfo("Pipeline running", "Please wait for the current run to finish.")
            return

        display_source = f"Camera {source}" if isinstance(source, int) else os.path.abspath(str(source))
        self.selected_source_var.set(f"Active source: {display_source}")
        self._append_log(f"Starting pipeline for {display_source}")

        self.running = True
        self.progress.start(8)
        self._reset_stage_status()
        self._reset_stage_previews()
        self._set_buttons_state("disabled")

        try:
            max_frames = self.max_frames_var.get()
        except tk.TclError:
            max_frames = 0
        max_frames_to_use = max_frames if max_frames > 0 else None

        save_outputs = self.save_var.get()
        frames_dir = self.frames_dir_var.get().strip() or None
        seg_dir = self.seg_dir_var.get().strip() or None
        if not save_outputs:
            frames_dir = None
            seg_dir = None

        def callback(step: str, message: str) -> None:
            self.ui_queue.put(("progress", step, message))

        def worker() -> None:
            try:
                result = run_classification_pipeline(
                    video_source=source,
                    max_frames=max_frames_to_use,
                    save_intermediate=save_outputs,
                    frames_dir=frames_dir,
                    seg_dir=seg_dir,
                    progress_callback=callback,
                )
                self.ui_queue.put(("result", result))
            except Exception as exc:  # noqa: BLE001
                self.ui_queue.put(("error", str(exc)))
            finally:
                self.ui_queue.put(("done", None))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------
    def _process_queue(self) -> None:
        try:
            while True:
                event = self.ui_queue.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, step, message = event
                    self._handle_progress(str(step), str(message))
                elif kind == "result":
                    _, result = event
                    self._display_result(result)  # type: ignore[arg-type]
                elif kind == "error":
                    _, message = event
                    self._handle_error(str(message))
                elif kind == "done":
                    self._finalize_run()
        except queue.Empty:
            pass
        finally:
            self.root.after(80, self._process_queue)

    def _handle_progress(self, step: str, message: str) -> None:
        label = STAGE_TITLES.get(step, step.title())
        if step in self.stage_vars:
            self.stage_vars[step].set(f"{label}: {message}")
        else:
            self.stage_vars["classification"].set(f"{label}: {message}")
        self._append_log(f"{label} - {message}")

    def _display_result(self, result: PipelineResult) -> None:  # type: ignore[name-defined]
        metadata = result.metadata
        metadata_text = (
            f"Source: {metadata.source}\n"
            f"Frames processed: {metadata.processed_frames}\n"
            f"Resolution: {metadata.resolution[0]}x{metadata.resolution[1]}\n"
            f"FPS: {metadata.fps if metadata.fps else 'unknown'}"
        )
        self.metadata_var.set(metadata_text)

        classification = result.classification
        votes_text = ", ".join(f"{key}: {value}" for key, value in classification.votes.items())
        stats = classification.statistics
        stats_text = (
            f"Mean speed: {format_stat(stats.get('speed', float('nan')))} px/frame\n"
            f"Mean displacement: {format_stat(stats.get('disp', float('nan')))} px\n"
            f"Mean turn angle: {format_stat(stats.get('angle', float('nan')))} deg"
        )

        self.classification_var.set(classification.label)
        self.votes_var.set(f"Votes - {votes_text if votes_text else '-'}")
        self.statistics_summary_var.set(stats_text)

        if result.bgr_frames:
            self.stage_views["preprocess"].update_image(result.bgr_frames[0])
            self.stage_vars["preprocess"].set(f"Preprocessing: {metadata.processed_frames} frames extracted")
        else:
            self.stage_views["preprocess"].show_placeholder("No frames captured")

        self.stage_views["segmentation"].update_image(result.segmentation_preview)
        self.stage_vars["segmentation"].set(f"Segmentation: {len(result.masks)} masks ready")

        self.stage_views["tracking"].update_image(result.tracking_preview)
        tracked_cells = 0
        if not result.linked.empty and "particle" in result.linked:
            tracked_cells = int(result.linked["particle"].nunique())
        self.stage_vars["tracking"].set(f"Tracking: {tracked_cells} trajectories")

        self.stage_vars["classification"].set(f"Classification: {classification.label}")

        self._update_features_table(result.features)
        if classification.label == "Insufficient Data":
            self._append_log("Warning: insufficient trajectories for confident classification.")
        else:
            self._append_log(f"Completed classification: {classification.label}")

        if result.frames_dir:
            self._append_log(f"Frames saved to: {os.path.abspath(result.frames_dir)}")
        if result.seg_dir:
            self._append_log(f"Masks saved to: {os.path.abspath(result.seg_dir)}")

    def _update_features_table(self, features: pd.DataFrame) -> None:
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)
        if features.empty:
            self.tracked_cells_var.set("Tracked cells: 0")
            return
        self.tracked_cells_var.set(f"Tracked cells: {len(features)}")
        preview = features.sort_values("particle").head(200)
        for _, row in preview.iterrows():
            self.features_tree.insert(
                "",
                "end",
                values=(
                    int(row["particle"]),
                    f"{row['speed']:.2f}",
                    f"{row['disp']:.2f}",
                    f"{row['angle']:.2f}",
                ),
            )

    def _handle_error(self, message: str) -> None:
        self._append_log(f"Error: {message}")
        messagebox.showerror("Pipeline error", message)

    def _finalize_run(self) -> None:
        self.running = False
        self.progress.stop()
        self._set_buttons_state("normal")
        self._append_log("Pipeline run finished.")

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _toggle_save_options(self) -> None:
        state = "normal" if self.save_var.get() else "disabled"
        self.frames_dir_entry.configure(state=state)
        self.seg_dir_entry.configure(state=state)

    def _reset_stage_status(self) -> None:
        for key, var in self.stage_vars.items():
            var.set(f"{STAGE_TITLES.get(key, key.title())}: Waiting")

    def _reset_stage_previews(self) -> None:
        for view in self.stage_views.values():
            view.show_placeholder()

    def _clear_results(self) -> None:
        if self.running:
            messagebox.showinfo("Pipeline running", "Cannot clear results while the pipeline is running.")
            return
        self.metadata_var.set("No video processed yet.")
        self.classification_var.set("No classification yet.")
        self.votes_var.set("Votes: -")
        self.statistics_summary_var.set("Speed / displacement / angle will appear here.")
        self.tracked_cells_var.set("Tracked cells: 0")
        self._reset_stage_status()
        self._reset_stage_previews()
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._append_log("Dashboard cleared.")

    def _set_buttons_state(self, state: str) -> None:
        for widget in (self.load_button, self.analyze_video_button, self.analyze_live_button, self.clear_button):
            widget.configure(state=state)
        if state == "normal":
            self.clear_button.configure(state="normal")

    def _on_close(self) -> None:
        if self.running:
            if not messagebox.askyesno("Quit", "The pipeline is still running. Exit anyway?"):
                return
        self.root.destroy()

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = CellClassifierApp()
    app.run()


if __name__ == "__main__":
    main()
