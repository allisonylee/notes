import cv2
import numpy as np
 
 
def detect_board(
    frame,
    min_area_fraction=0.10,
    max_area_fraction=0.97,
    approx_epsilon=0.02,
    output_size=(1280, 720),
    debug=False,
):
    """
    Detect a whiteboard or blackboard in a single BGR frame.
 
    Returns a dict:
        found        (bool)
        corners      (np.ndarray shape (4,2), TL/TR/BR/BL order) or None
        confidence   (float 0–1)
        warped       (BGR np.ndarray, perspective-corrected) or None
        debug_frame  (BGR np.ndarray annotated) or None — only when debug=True
    """
    h, w = frame.shape[:2]
    frame_area = h * w
 
    # 1. Pre-process
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=5,
    )
    edges = cv2.Canny(thresh, 30, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
 
    # 2. Find the largest quadrilateral in a sensible area range
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
 
    best_quad = None
    best_area = 0.0
 
    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        area_frac = area / frame_area
 
        if area_frac < min_area_fraction:
            break
        if area_frac > max_area_fraction:
            continue
 
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_epsilon * peri, True)
 
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best_quad = approx.reshape(4, 2).astype(np.float32)
            best_area = area
            break
 
    if best_quad is None:
        return {
            "found": False,
            "corners": None,
            "confidence": 0.0,
            "warped": None,
            "debug_frame": _draw_debug(frame, edges, None, None) if debug else None,
        }
 
    # 3. Order corners TL → TR → BR → BL
    corners = _order_corners(best_quad)
 
    # 4. Perspective warp
    out_w, out_h = output_size
    dst = np.array([
        [0,       0      ],
        [out_w-1, 0      ],
        [out_w-1, out_h-1],
        [0,       out_h-1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, M, (out_w, out_h))
 
    # 5. Confidence heuristic
    area_frac = best_area / frame_area
    confidence = float(np.clip(1.0 - abs(area_frac - 0.45) / 0.45, 0.1, 1.0))
 
    return {
        "found": True,
        "corners": corners,
        "confidence": confidence,
        "warped": warped,
        "debug_frame": _draw_debug(frame, edges, best_quad, corners) if debug else None,
    }
 
 
def _order_corners(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],    # top-left
        pts[np.argmin(diff)], # top-right
        pts[np.argmax(s)],    # bottom-right
        pts[np.argmax(diff)], # bottom-left
    ], dtype=np.float32)
 
 
def _draw_debug(frame, edges, quad, corners):
    out = frame.copy()
    edge_color = np.zeros_like(out)
    edge_color[edges > 0] = (200, 100, 0)
    out = cv2.addWeighted(out, 0.8, edge_color, 0.4, 0)
 
    if quad is not None:
        cv2.drawContours(out, [quad.astype(np.int32)], -1, (0, 255, 0), 3)
 
    if corners is not None:
        for (x, y), label in zip(corners.astype(int), ["TL", "TR", "BR", "BL"]):
            cv2.circle(out, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(out, label, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out
 