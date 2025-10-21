# Cell Movement Classifier GUI

This minimal app wraps your existing pipeline (preprocess → segment → track → features → classify) in a simple GUI so you can:
- Load a video **or** record 10 seconds from your laptop webcam
- Run the pipeline
- See the **prediction**: *Normal* vs *Cancer*
- Inspect saved outputs (frames, segmentation masks, tracking.csv, features.csv)

## Project layout

```
your_folder/
├─ src/                      # your existing code (as in the zip)
│  ├─ preprocess.py
│  ├─ segment.py
│  ├─ track.py
│  ├─ features.py
│  └─ metadata.py
├─ cell_gui_app.py           # the GUI app
└─ ResultsGUI/               # created on first run with per-video subfolders
```

## How to run

1) Create/activate a Python 3.10–3.11 environment.
2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Put your `src/` folder (from your zip) next to `cell_gui_app.py`.
4) Launch:

```bash
python cell_gui_app.py
```

## Dependencies

See `requirements.txt`. The UI uses Tkinter which ships with most Python distributions. If you use conda/miniconda on Windows and Tk is missing, install `tk` once:

```bash
conda install tk
```

## Notes

- The thresholds used for classification come from your scripts:
  - speed > 49, angle > 80, displacement < 21 → **Cancer**
  - otherwise → **Normal**
- All outputs for each video are written under `ResultsGUI/<video_stem>/`:
  - `frames/` (preprocessed PNGs)
  - `segmentation/` (binary masks)
  - `tracking.csv` (trackpy output)
  - `features.csv` (aggregated motion stats)

If you want to tweak thresholds or add a probabilistic score, open `cell_gui_app.py` and modify the **[thresholds]** section.
