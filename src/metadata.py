import cv2
import os
import pandas as pd
from pprint import pprint


def extract_metadata(video_path):
    """Extract basic metadata from a given video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    metadata = {
        "Video Name": os.path.basename(video_path),
        "Video Path": video_path,
        "Resolution": f"{width}x{height}",
        "Frame Rate (fps)": round(fps, 2),
        "Frame Count": frame_count,
        "Duration (sec)": round(duration, 2)
    }
    return metadata


if __name__ == "__main__":
    video_path = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Dataset\Cancer1.mp4"
    output_csv = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\cancer1_metadata.csv"


    info = extract_metadata(video_path)
    if info:
        print("\n[VIDEO METADATA]")
        pprint(info)

        # Save metadata to CSV for reference
        output_csv = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\normal_metadata.csv"
        pd.DataFrame([info]).to_csv(output_csv, index=False)
        print(f"\n[INFO] Metadata saved to: {output_csv}")
    else:
        print("[ERROR] Metadata extraction failed.")
