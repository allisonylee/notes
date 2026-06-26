import cv2
import time
from typing import Generator, Optional
 
 
def capture_frames(
    camera_index: int = 0,
    target_fps: Optional[float] = None,
    resolution: Optional[tuple[int, int]] = None,
    max_frames: Optional[int] = None,
) -> Generator[tuple[int, float, cv2.typing.MatLike], None, None]:
    """
    Capture frames from a live webcam as a generator.
 
    Yields (frame_number, timestamp, frame) tuples so the caller can
    plug in any analysis logic without worrying about camera lifecycle.
 
    Args:
        camera_index:  OpenCV device index (0 = default webcam).
        target_fps:    Cap capture rate to this many frames per second.
                       None means capture as fast as the camera allows.
        resolution:    (width, height) to request from the camera.
                       None keeps the camera's native resolution.
        max_frames:    Stop after this many frames. None runs indefinitely
                       until the caller breaks or the camera disconnects.
 
    Yields:
        frame_number:  Zero-based frame counter.
        timestamp:     time.monotonic() value at capture (seconds).
        frame:         BGR numpy array (H x W x 3), ready for cv2 calls.
 
    Raises:
        RuntimeError:  If the camera cannot be opened.
 
    Example:
        for frame_num, ts, frame in capture_frames(target_fps=10):
            result = detect_board(frame)
            if result.found:
                break
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}")
 
    if resolution is not None:
        width, height = resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
 
    min_interval = (1.0 / target_fps) if target_fps else 0.0
    frame_number = 0
    last_capture_time = 0.0
 
    try:
        while True:
            now = time.monotonic()
            elapsed = now - last_capture_time
 
            # Throttle to target_fps if requested
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

def preview_webcam(camera_index: int = 0, window_title: str = "Webcam Preview"):
    """
    Show a live webcam window. Press 'q' to quit.
    Useful for verifying the camera works before wiring in analysis.
    """
    for _, _, frame in capture_frames(camera_index=camera_index):
        cv2.imshow(window_title, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

def main():
    preview_webcam()
    return




if __name__ == "__main__":
    main()