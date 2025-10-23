import cv2
import numpy as np

def detect_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=150, maxLineGap=20)
    return lines

def crop_court(frame, lines):
    if lines is None:
        return frame
    all_x, all_y = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        all_x += [x1, x2]
        all_y += [y1, y2]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    return frame[y_min:y_max, x_min:x_max]
