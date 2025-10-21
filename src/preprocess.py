import cv2
import os

def preprocess_video(video_path, output_dir):
    """Convert video into enhanced grayscale frames with denoising,
    background subtraction, and contrast enhancement."""

    # --- Prepare output directory ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Open the video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return

    # ---------------------------------------------------------------------
    # Initialize Background Subtractor
    #
    # history = 50 → how many previous frames are used to build the background model.
    #   In microscopy videos, illumination and background changes are gradual.
    #   A value of 50 allows the algorithm to learn a stable background
    #   over roughly 50 frames (~7 seconds in our video), which balances
    #   adaptability and noise reduction.
    #
    # varThreshold = 25 → the threshold on the squared Mahalanobis distance
    #   between the pixel and the background model to decide if it’s foreground.
    #   Smaller values detect only strong motion; larger values detect even
    #   subtle cell movements but risk more noise.
    #   25 is a balanced mid-range value that captures slow-moving cells
    #   while suppressing background fluctuations.
    #
    # detectShadows=False → disables shadow detection to avoid misclassifying
    #   faint cell edges as shadows (common in microscope lighting).
    # ---------------------------------------------------------------------
    fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Convert to grayscale (color info is unnecessary for cell videos)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Denoising (remove random intensity noise and smooth texture)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 3. Background subtraction (highlight moving or changing regions)
        fgmask = fgbg.apply(denoised)

        # 4. Contrast enhancement (equalize histogram for clearer visibility)
        enhanced = cv2.equalizeHist(fgmask)

        # 5. Save processed frame
        filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(filename, enhanced)
        frame_idx += 1

    cap.release()
    print(f"[INFO] Preprocessing completed — {frame_idx} frames saved in '{output_dir}'.")


if __name__ == "__main__":
    # --- Absolute paths ---
    if __name__ == "__main__":
        video_path = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Dataset\Cancer1.mp4"
        output_dir = r"C:\Users\shaha\PycharmProjects\cell_movement_analysis\Results\frames_cancer1"

    # --- Run preprocessing ---
    preprocess_video(video_path, output_dir)
