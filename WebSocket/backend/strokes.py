import time
import uuid
import cv2
import numpy as np
 
 
# ---------------------------------------------------------------------------
# Module-level state — persists across calls within a session
# ---------------------------------------------------------------------------
 
_prev_board = None   # last warped board frame we compared against
 
 
def reset():
    """Call this when a new session starts so stale state is cleared."""
    global _prev_board
    _prev_board = None
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def find_new_strokes(warped):
    """
    Compare the current warped board frame against the previous one and
    return any new strokes that appeared.
 
    Args:
        warped: BGR numpy array — the perspective-corrected board from detect_board().
 
    Returns:
        List of stroke dicts (may be empty). Each stroke:
        {
            "id":        str,
            "points":    [[x, y], ...],   # in warped-frame pixel coords
            "color":     [r, g, b],
            "thickness": int,
            "timestamp": float,
        }
    """
    global _prev_board
 
    if _prev_board is None:
        _prev_board = warped.copy()
        return []
 
    diff_mask = _diff_mask(warped, _prev_board)
    _prev_board = warped.copy()
 
    if diff_mask is None:
        return []
 
    strokes = _extract_strokes(warped, diff_mask)
    return strokes
 
 
# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
 
def _diff_mask(current, previous):
    """
    Return a binary mask of pixels that changed meaningfully between frames.
    Returns None if the frames are essentially identical.
    """
    # Work in grayscale for the diff, then threshold
    curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
 
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
 
    # Remove speckle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Fill small holes inside strokes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
 
    if cv2.countNonZero(mask) < 50:   # ignore tiny noise
        return None
 
    return mask
 
 
def _extract_strokes(warped, mask):
    """
    Find connected components in the diff mask and turn each one into a stroke.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    strokes = []
    ts = time.time()
 
    for label in range(1, num_labels):   # 0 is background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 30:   # skip tiny fragments
            continue
 
        component_mask = (labels == label).astype(np.uint8) * 255
        points = _skeleton_points(component_mask)
        color = _sample_color(warped, component_mask)
        thickness = _estimate_thickness(component_mask)
 
        strokes.append({
            "id": str(uuid.uuid4())[:8],
            "points": points,
            "color": color,
            "thickness": thickness,
            "timestamp": ts,
        })
 
    return strokes
 
 
def _skeleton_points(mask):
    """
    Return an ordered list of [x, y] points along the stroke centreline.
    Uses contour moments on horizontal slices for a simple, cheap approximation.
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return []
 
    y_min, y_max = int(ys.min()), int(ys.max())
 
    # Sample ~40 evenly-spaced rows and find the centroid x in each
    step = max(1, (y_max - y_min) // 40)
    points = []
    for y in range(y_min, y_max + 1, step):
        row_xs = xs[ys == y]
        if len(row_xs) == 0:
            continue
        x = int(row_xs.mean())
        points.append([x, y])
 
    return points
 
 
def _sample_color(warped, mask):
    """
    Return the median BGR color of pixels under the mask, as [r, g, b].
    """
    pixels = warped[mask > 0]   # shape (N, 3)
    if len(pixels) == 0:
        return [0, 0, 0]
    median = np.median(pixels, axis=0).astype(int).tolist()
    b, g, r = median
    return [r, g, b]
 
 
def _estimate_thickness(mask):
    """
    Estimate stroke width by dividing its area by its length (bounding-box height).
    """
    ys, _ = np.where(mask > 0)
    height = int(ys.max()) - int(ys.min()) + 1
    area = int(cv2.countNonZero(mask))
    thickness = max(1, area // max(height, 1))
    return int(thickness)