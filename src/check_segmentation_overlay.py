import cv2
import os
import numpy as np

# Paths to preprocessed frames and segmentation masks
frames_dir = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\frames"
masks_dir = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\segmentation"

# Sorted frame names to keep same order
frame_files = sorted(os.listdir(frames_dir))

for i in range(0, len(frame_files), max(1, len(frame_files)//5)):  # show ~5 overlays
    frame_path = os.path.join(frames_dir, frame_files[i])
    mask_path = os.path.join(masks_dir, frame_files[i])

    frame = cv2.imread(frame_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Convert mask to red overlay for visualization
    color_mask = np.zeros_like(frame)
    color_mask[:, :, 2] = mask  # put mask in red channel

    # Overlay mask (red) onto original frame
    overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

    cv2.imshow("Segmentation Overlay", overlay)
    print(f"Showing overlay for: {frame_files[i]}")
    key = cv2.waitKey(0)
    if key == 27:  # ESC to stop early
        break

cv2.destroyAllWindows()
