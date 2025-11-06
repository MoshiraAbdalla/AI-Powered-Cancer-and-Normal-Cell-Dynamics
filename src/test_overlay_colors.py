#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2

from cell_gui_app import (
    run_pipeline,
    build_segmentation_overlay,
    build_tracking_overlay,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample segmentation and tracking overlays to verify GUI colouring."
    )
    default_video = Path(__file__).resolve().parent / "Dataset" / "Cancer2.mp4"
    parser.add_argument(
        "--video",
        type=Path,
        default=default_video,
        help="Path to the video to analyse (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Results" / "overlay_demos",
        help="Directory where overlay images will be stored.",
    )
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    results_root = args.output.expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    overlay_dir = results_root / f"{video_path.stem}_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    counters = {"seg": 0}

    def preview_callback(evt):
        typ = evt.get("type")
        if typ == "seg_overlay":
            img = build_segmentation_overlay(evt["frame_path"], evt["mask_path"])
            if img is None or counters["seg"] >= 5:
                return
            out_path = overlay_dir / f"seg_overlay_{counters['seg']:03d}.png"
            cv2.imwrite(str(out_path), img)
            counters["seg"] += 1
        elif typ == "tracking_overlay":
            img = build_tracking_overlay(evt["frame_path"], evt["linked_df"])
            if img is None:
                return
            out_path = overlay_dir / "tracking_overlay.png"
            cv2.imwrite(str(out_path), img)

    def progress_callback(step, msg):
        print(f"[STEP {step}] {msg}")

    def log_callback(msg):
        print(msg)

    classification, votes, means, _ = run_pipeline(
        str(video_path),
        str(results_root),
        log_fn=log_callback,
        progress_fn=progress_callback,
        preview_fn=preview_callback,
    )

    mean_speed, mean_disp, mean_angle = means
    print("\nFinished. Diagnostics:")
    print(f"  Classification : {classification}")
    print(f"  Votes          : {votes}")
    print(f"  Mean Speed     : {mean_speed:.2f}")
    print(f"  Mean Disp      : {mean_disp:.2f}")
    print(f"  Mean Angle     : {mean_angle:.2f}")
    print(f"\nOverlay images saved to: {overlay_dir}")


if __name__ == "__main__":
    main()
