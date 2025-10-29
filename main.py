import cv2
from utils.detection import BadmintonDetector


def main():
    detector = BadmintonDetector()

    # Load video
    cap = cv2.VideoCapture("videos/00001.mp4")
    if not cap.isOpened():
        print("Error opening video file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect players and shuttle
        players = detector.detect_players(frame)
        shuttle = detector.detect_shuttle(frame)

        # Draw detections
        frame = detector.draw_detections(frame, players, shuttle)

        # Display frame
        cv2.imshow("Badminton Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
