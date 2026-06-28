import time
import cv2
 
 
def capture_frames(camera_index=0, target_fps=None, resolution=None, max_frames=None):
    """
    Capture frames from a live webcam as a generator.
 
    Yields (frame_number, timestamp, frame) tuples.
 
    Args:
        camera_index:  OpenCV device index (0 = default webcam).
        target_fps:    Cap capture rate. None = as fast as camera allows.
        resolution:    (width, height) to request. None = camera native.
        max_frames:    Stop after this many frames. None = run indefinitely.
 
    Raises:
        RuntimeError: If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}")
 
    if resolution is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
 
    min_interval = (1.0 / target_fps) if target_fps else 0.0
    frame_number = 0
    last_capture_time = 0.0
 
    try:
        while True:
            now = time.monotonic()
            elapsed = now - last_capture_time
 
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
                now = time.monotonic()
 
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame — camera may have disconnected.")
                break
 
            last_capture_time = now
            yield frame_number, now, frame
 
            frame_number += 1
            if max_frames is not None and frame_number >= max_frames:
                break
    finally:
        cap.release()
 