from ultralytics import YOLO
import cv2
import numpy as np

# ===========================
# 🔹 Load YOLOv8 pose model for badminton court
# ===========================
model = YOLO(r"C:\badm\runs\pose\train19\weights\best.pt")  # update path if needed

# ===========================
# 🔹 Open video
# ===========================
video_path = r"C:\badm\badminton-scoring\videos\00353.mp4"
cap = cv2.VideoCapture(video_path)

# ===========================
# 🔹 Helper Functions
# ===========================
def get_ordered_corners(points):
    """
    Compute convex hull and return ordered corners TL, TR, BR, BL
    points: numpy array of shape [n, 2]
    """
    hull = cv2.convexHull(points.astype(np.float32))
    hull = hull[:, 0, :]  # remove extra dimension

    s = hull.sum(axis=1)
    diff = np.diff(hull, axis=1)

    TL = hull[np.argmin(s)]
    BR = hull[np.argmax(s)]
    TR = hull[np.argmin(diff)]
    BL = hull[np.argmax(diff)]

    return np.array([TL, TR, BR, BL], dtype=np.int32)

def draw_dashed_line(img, pt1, pt2, color=(0, 0, 255), thickness=3, dash_length=10):
    pt1 = np.array(pt1, dtype=int)
    pt2 = np.array(pt2, dtype=int)
    line_len = np.linalg.norm(pt2 - pt1)
    dashes = int(line_len // dash_length)
    for i in range(dashes):
        start = pt1 + (pt2 - pt1) * (i / dashes)
        end = pt1 + (pt2 - pt1) * ((i + 0.5) / dashes)
        cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)

# Ratio to shift net toward Player 2 (0.5 = center, <0.5 → shift up, >0.5 → shift down)
shift_ratio = 0.33  # Player 1 gets more area

# ===========================
# 🔹 Process Video
# ===========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.35, verbose=False)
    annotated_frame = frame.copy()

    for result in results:
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            continue

        keypoints = result.keypoints.xy[0].cpu().numpy()
        if hasattr(result.keypoints, 'conf'):
            confs = result.keypoints.conf[0].cpu().numpy()
            keypoints = keypoints[confs > 0.3]

        if len(keypoints) < 4:
            continue

        TL, TR, BR, BL = get_ordered_corners(keypoints)
        outer_corners = np.array([TL, TR, BR, BL], np.int32)

        # Draw court boundary
        cv2.polylines(annotated_frame, [outer_corners], isClosed=True, color=(0, 255, 0), thickness=3)

        # ===========================
        # 🔹 Horizontal net line shifted toward Player 2
        # Extend net fully across court width
        net_y = int(BL[1] * shift_ratio + TL[1] * (1 - shift_ratio))
        net_left = (min(TL[0], TR[0], BL[0], BR[0]), net_y)
        net_right = (max(TL[0], TR[0], BL[0], BR[0]), net_y)
        draw_dashed_line(annotated_frame, net_left, net_right, color=(0, 0, 255), thickness=3, dash_length=15)

        # ===========================
        # 🔹 Label Player 1 and Player 2 courts
        p1_text_pos = (BL[0] + 30, (BL[1] + net_y) // 2)
        p2_text_pos = (TL[0] + 30, (TL[1] + net_y) // 2)
        cv2.putText(annotated_frame, "PLAYER 1 COURT", p1_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.putText(annotated_frame, "PLAYER 2 COURT", p2_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Print court points and net line
        print("Court points (TL, TR, BR, BL):", outer_corners)
        print("Horizontal net line (shifted):", net_left, net_right)

    # Display result
    cv2.imshow("Badminton Court Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
