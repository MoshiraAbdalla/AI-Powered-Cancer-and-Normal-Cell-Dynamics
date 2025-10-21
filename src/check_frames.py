import cv2
import os

# Path to your frames
frames_dir = r"./Results/frames"

# List all frames
frame_files = sorted(os.listdir(frames_dir))

# Show the first few frames
for i in range(0, len(frame_files), max(1, len(frame_files)//5)):  # show 5 evenly spaced frames
    frame_path = os.path.join(frames_dir, frame_files[i])
    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Preprocessed Frame", img)
    print(f"Showing: {frame_files[i]}")
    key = cv2.waitKey(0)
    if key == 27:  # Press ESC to exit early
        break

cv2.destroyAllWindows()
