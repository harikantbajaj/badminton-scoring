from ultralytics import YOLO
import cv2

# Load YOLO models
player_model = YOLO('models/yolov8n.pt')      # Detect players
shuttle_model = YOLO('models/shuttle.pt')     # Detect shuttlecock

# Load video
cap = cv2.VideoCapture('videos/sample_video.mp4')

# Optional: define court ROI manually (example)
x1, y1, x2, y2 = 100, 50, 1180, 620

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop court area
    court_frame = frame[y1:y2, x1:x2]

    # Player detection
    players = player_model(court_frame)[0]
    frame_players = players.plot()

    # Shuttle detection
    shuttle = shuttle_model(court_frame)[0]
    frame_shuttle = shuttle.plot()

    # Combine visualization
    combined_frame = cv2.addWeighted(frame_players, 0.7, frame_shuttle, 0.7, 0)

    # Display
    cv2.imshow('Phase 1 - Object Detection', combined_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, player_model_path, shuttle_model_path):
        self.player_model = YOLO(player_model_path)
        self.shuttle_model = YOLO(shuttle_model_path)

    def detect(self, frame):
        # Run inference
        players = self.player_model.predict(frame, classes=[0], conf=0.5, verbose=False)  # class 0 = person
        shuttle = self.shuttle_model.predict(frame, conf=0.3, verbose=False)

        # Extract boxes
        player_boxes = players[0].boxes.xyxy.cpu().numpy() if players else []
        shuttle_boxes = shuttle[0].boxes.xyxy.cpu().numpy() if shuttle else []

        return player_boxes, shuttle_boxes

    def draw_boxes(self, frame, player_boxes, shuttle_boxes):
        for (x1, y1, x2, y2) in player_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, "Player", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for (x1, y1, x2, y2) in shuttle_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, "Shuttle", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame
