# Cell  Classifier GUI 

`GUI_diff.py` is the "pro" Tkinter interface for the cell-movement pipeline. It exposes the full preprocess -> segment -> track -> feature -> classify workflow, adds live previews for every stage, records detailed logs, and introduces diffusion-entropy assisted voting so clinicians can see *why* a sample looks normal or cancerous.

---

## Highlights
- **Guided, color-coded UI** with HiDPI scaling, tabbed previews (preprocess, segmentation, tracking, trajectories, diffusion entropy) and dedicated classification metrics.
- **One-click workflow**: open an MP4/AVI (or record a clip from the laptop webcam) and run the entire pipeline inside the GUI thread.
- **Rich outputs**: frames, segmentation masks, linked tracks, feature CSVs, trajectory overlays, diffusion-entropy plots, and the final prediction are saved in a per-video folder under `src/Results/`.
- **Transparent voting**: mean speed, displacement, turning angle, persistence, and the DEA slope each cast a vote (Normal vs Cancer). Thresholds are clearly surfaced for auditing.
- **Built-in logging & shortcuts**: real-time log panel plus `Ctrl+R` (run), `Ctrl+L` (start live), `Ctrl+Shift+L` (stop live), and quick access to the results directory.

---

## Requirements
- Python 3.10 or 3.11 (Tkinter ships with the standard installer; on conda use `conda install tk` if needed).
- Packages listed in `requirements.txt`:
  - `opencv-python`, `numpy`, `pandas`, `pillow`, `scikit-image`, `trackpy`, plus the standard library modules bundled with Python.
- A webcam is optional but required for the **Start Live** capture workflow.

Create a virtual environment and install the dependencies once:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start
1. Open a terminal in `src/` and activate your environment.
2. Launch the GUI:
   ```bash
   python GUI_diff.py
   ```
3. Either click **Open Video** and pick an MP4/AVI/MOV/MKV file, or click **Start Live** to capture a new clip (press **Stop Live** once you have ~10 s recorded).
4. Hit **Run Classification** (or press `Ctrl+R`). The progress bar will step through Frames -> Segmentation -> Tracking -> Features -> Classification while the log panel streams pipeline messages.
5. Explore the tabbed previews to inspect intermediate results, then review the **Classification** tab for the final verdict and per-metric votes.

All artifacts are written to `src/Results/<video_stem>/` automatically; reuse those folders for reports or further research.

---

## What Each Tab Shows
| Tab | What you see | Source files |
| --- | --- | --- |
| **1) Preprocess** | Representative grayscale frame sampled while frames are extracted. | `Results/<video>/frames/frame_XXX.png` |
| **2) Segmentation** | Overlay of the current binary mask on the grayscale frame. | `frames/` + `segmentation/` PNGs |
| **3) Tracking (Overlay)** | Last frame with colored tracks for every linked cell trajectory. | `tracking_overlay_last.png` |
| **4) Trajectories Plot** | Dense Matplotlib plot of every trajectory with direction arrows. | `trajectories_plot.png` |
| **5) Diffusion Entropy** | Log-log diffusion entropy curve and fitted slope. | `diffusion_entropy_dense.png` |
| **6) Classification** | Final label, vote breakdown, mean speed/dispersion/angle, persistence, DEA slope, and log access. | `features.csv`, DEA output |

Use the **Open Results Folder** button to jump straight to these files in Explorer/Finder.

---

## Output Layout
Every run creates (or reuses) a folder: `src/Results/<video_stem>/`

```
frames/                    # grayscale PNG sequence
segmentation/              # binary masks (same filenames)
tracks_linked.csv          # per-frame trackpy output with particle IDs
tracking_overlay_last.png  # BGR overlay saved from the final frame
trajectories_plot.png      # Matplotlib plot covering all tracks
features.csv               # per-particle speed/disp/angle/persistence stats
diffusion_entropy_dense.png# S(t) vs time with linear fit and slope
```

Live captures are stored under `src/Results/Live/live_<timestamp>.mp4` and automatically selected for inference once recording stops.

---

## Classification & Voting Logic
`GUI_diff.py` computes cell-wise motion features, aggregates them, and casts votes:

| Feature (video-level mean) | Threshold | Vote |
| --- | --- | --- |
| Speed (px/s) | `> 46.54` | Cancer if above threshold, else Normal |
| Displacement (px) | `> 20.78` | Normal if above threshold, else Cancer |
| Turn angle (deg) | `> 74.82` | Cancer if above threshold, else Normal |
| Persistence (D/L) | `> 0.50` | Normal if above threshold, else Cancer |
| Diffusion Entropy slope (`delta`) | `> 0.50` | Cancer if DEA fit is available and slope exceeds threshold |

The label becomes **Cancer Cell** when Cancer votes outnumber Normal votes, otherwise **Normal Cell**. The detailed values and votes are shown in the **Classification** tab and mirrored in the log.

Tweak thresholds inside `run_pipeline()` (see lines ~575-655) if you recalibrate on new datasets. For longer clips you can also change the `search_range`, `memory`, or `tail_len` constants to better match cell density.

---

## Live Recording Workflow
1. Click **Start Live** (or press `Ctrl+L`). The GUI opens `cv2.VideoCapture(0)` and begins writing an MP4 under `Results/Live/` while previewing the grayscale feed in the **Preprocess** tab.
2. Collect ~10 seconds, then click **Stop Live** (`Ctrl+Shift+L`). The file is closed, the entry field updates with the saved path, and the status text switches to "Live recording saved. Ready to run classification."
3. Run classification as usual; the live capture stays on disk for later auditing.

Stop live before running the pipeline - `GUI_diff.py` keeps capture and classification on separate threads to avoid freezing the UI.

---

## Tips & Customization
- **HiDPI/Scaling:** Adjust `enable_hidpi(scaling=...)` near the top if fonts or previews look too large/small on your monitor.
- **DEA requirements:** The diffusion entropy plot needs enough trajectories; if the slope is `n/a`, collect a longer clip or increase magnification.
- **GPU vs CPU:** Everything runs on CPU via OpenCV/NumPy/trackpy. For large videos, close other apps to free RAM before launching.
- **Clean re-runs:** Delete the relevant folder under `src/Results/` if you want to reprocess a video from scratch.
- **Logs:** Use the log panel to diagnose issues (e.g., segmentation failures or DEA warnings). Messages are mirrored in the console running the script.

`GUI_diff.py` is self-contained - no extra config files are required. If you extend the pipeline (e.g., replacing segmentation or adding probabilistic scores), surface your new metrics inside `run_pipeline()` so the GUI can display them without further wiring.
