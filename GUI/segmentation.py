# GUI/segmentation.py
import cv2
import numpy as np
import time

MIN_SEG_CONFIDENCE = 0.95
CENTER_TOL = 40

def approximate_quad(contour, desired_points=4, initial_scale=0.01, step_scale=0.005, max_scale=0.1):
    arc_len = cv2.arcLength(contour, True)
    scale = initial_scale
    approx = cv2.approxPolyDP(contour, scale * arc_len, True)
    while len(approx) != desired_points and scale < max_scale:
        scale += step_scale
        approx = cv2.approxPolyDP(contour, scale * arc_len, True)
    return approx

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def compute_focus_measure(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_segmentation(frame, seg_model):
    start_time = time.perf_counter()
    try:
        results = seg_model(frame, conf=0.7)
        result = results[0]
        if result.masks is not None and result.masks.xy is not None:
            for polygon in result.masks.xy:
                confidence = result.masks.conf[0] if hasattr(result.masks, 'conf') else 1.0
                if confidence < MIN_SEG_CONFIDENCE:
                    continue
                contour = polygon.astype(np.float32).reshape(-1, 1, 2)
                quad = approximate_quad(contour)
                if quad is None or len(quad) != 4:
                    rect = cv2.minAreaRect(contour)
                    quad = cv2.boxPoints(rect)
                    quad = quad.astype(np.int32)
                else:
                    quad = quad.reshape(-1, 2).astype(np.int32)
                end_time = time.perf_counter()
                print(f"Segmentation step took {end_time - start_time:.3f} seconds")
                return quad
    except Exception as e:
        print("Segmentation error:", e)
    end_time = time.perf_counter()
    print(f"Segmentation step took {end_time - start_time:.3f} seconds (no valid detection)")
    return None

def is_card_centered(quad, frame, tol=CENTER_TOL):
    if quad is None:
        return False
    M = cv2.moments(quad)
    if M["m00"] == 0:
        return False
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    (h, w) = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2
    dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    return dist <= tol