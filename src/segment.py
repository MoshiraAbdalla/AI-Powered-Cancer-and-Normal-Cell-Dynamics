import cv2
import os
import numpy as np

def segment_cells(frames_dir, output_dir):
    """
    Perform cell segmentation on preprocessed frames.
    This version uses adaptive Gaussian blur + Otsu thresholding
    to create binary masks highlighting each cell.
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted(os.listdir(frames_dir))
    total = 0

    for file in frame_files:
        if not file.endswith(".png"):
            continue
        path = os.path.join(frames_dir, file)

        # --- Read frame ---
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # --- Step 1: Smooth slightly to remove small specks ---
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        # --- Step 2: Apply Otsu’s threshold ---
        # Automatically finds the intensity threshold that best separates foreground from background.
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- Step 3: Morphological cleaning ---
        # Remove isolated noise and small pixel groups that are not real cells.
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # --- Save the mask ---
        cv2.imwrite(os.path.join(output_dir, file), cleaned)
        total += 1

    print(f"[INFO] Segmentation complete — {total} masks saved to '{output_dir}'.")


if __name__ == "__main__":
    # Absolute paths for your setup
    if __name__ == "__main__":
        frames_dir = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\frames_cancer1"
        output_dir = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\segmentation_cancer1"

    segment_cells(frames_dir, output_dir)
