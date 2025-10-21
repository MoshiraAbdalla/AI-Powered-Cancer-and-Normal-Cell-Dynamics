import cv2
import os
import numpy as np

# --- Paths to preprocessed frames and segmentation masks for Cancer1 ---
frames_dir = r"./Results/frames_cancer1"
masks_dir = r"./Results/segmentation_cancer1"

# --- Sorted frame names to keep the same order ---
frame_files = sorted(os.listdir(frames_dir))

# --- Display around 5 overlays for quick validation ---
for i in range(0, len(frame_files), max(1, len(frame_files)//5)):
    frame_path = os.path.join(frames_dir, frame_files[i])
    mask_path = os.path.join(masks_dir, frame_files[i])

    # Load the original frame and its segmentation mask
    frame = cv2.imread(frame_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # --- Ensure both are loaded correctly ---
    if frame is None or mask is None:
        print(f"[WARNING] Could not load frame or mask: {frame_files[i]}")
        continue

    # --- Convert mask to red overlay for visualization ---
    color_mask = np.zeros_like(frame)
    color_mask[:, :, 2] = mask  # Add mask in red channel (BGR → R)

    # --- Overlay the mask (red) on the original frame ---
    overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

    # --- Display overlay ---
    cv2.imshow("Segmentation Overlay - Cancer1", overlay)
    print(f"Showing overlay for: {frame_files[i]}")

    # Press ESC to exit early, or any key to continue
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()
