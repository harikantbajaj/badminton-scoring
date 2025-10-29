from ultralytics import YOLO
import cv2


class BadmintonDetector:
    def __init__(
        self,
        player_model_path="models/yolov8n.pt",
        shuttle_model_path="models/shuttle.pt",
    ):
        self.player_model = YOLO(player_model_path)
        self.shuttle_model = YOLO(shuttle_model_path)

    def detect_players(self, frame):
        # Run inference for players
        players = self.player_model.predict(
            frame, classes=[0], conf=0.5, verbose=False
        )  # class 0 = person
        # Extract boxes
        player_boxes = players[0].boxes.xyxy.cpu().numpy() if players else []
        return player_boxes

    def detect_shuttle(self, frame):
        # Run inference for shuttle
        shuttle = self.shuttle_model.predict(frame, conf=0.3, verbose=False)
        # Extract boxes
        shuttle_boxes = shuttle[0].boxes.xyxy.cpu().numpy() if shuttle else []
        return shuttle_boxes

    def draw_detections(self, frame, player_boxes, shuttle_boxes):
        for x1, y1, x2, y2 in player_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Player",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        for x1, y1, x2, y2 in shuttle_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "Shuttle",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        return frame
