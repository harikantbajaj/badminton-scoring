import cv2
from ultralytics import YOLO
from collections import deque
import torch
import os

# ===========================
# 🔹 GPU Check
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🟢 Using device: {device}")

# ===========================
# 🔹 Load YOLO Shuttle Model
# ===========================
model_path = r"C:\badm\runs\detect\train8\weights\best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found: {model_path}")

model = YOLO(model_path)
print("✅ YOLO Shuttlecock Model Loaded")

# ===========================
# 🔹 Load Video or Webcam
# ===========================
video_path = r"C:\badm\badminton-scoring\videos\00777.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video: {video_path}")

# ===========================
# 🔹 Tracking Setup
# ===========================
shuttle_traj = deque(maxlen=30)
frame_count = 0

# ===========================
# 🔹 Main Detection Loop
# ===========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Video ended or not readable")
        break

    frame_count += 1

    # Run YOLO detection
    results = model(frame, conf=0.45, device=device)

    # Extract shuttle positions
    shuttle_positions = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            shuttle_positions.append((cx, cy))
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

    # Track trajectory
    if shuttle_positions:
        shuttle_traj.append(shuttle_positions[0])

    # Draw shuttle trajectory
    for i in range(1, len(shuttle_traj)):
        cv2.line(frame, shuttle_traj[i - 1], shuttle_traj[i], (0, 255, 255), 2)

    # Display detection frame
    cv2.putText(frame, "🏸 Shuttlecock Tracking", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Shuttle Detection", frame)

    # Slightly slower playback
    if cv2.waitKey(20) & 0xFF == ord('q'):  # 30 ms delay → ~33 FPS
        break

# ===========================
# 🔹 Cleanup
# ===========================
cap.release()
cv2.destroyAllWindows()
print("✅ Shuttle Detection Completed")



