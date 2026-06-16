from ultralytics import YOLO
import cv2


class PlayerDetector:
    """Detects and tracks players and sports balls in video frames using YOLO."""

    def __init__(self, model_name="yolov8n.pt", confidence=0.4):
        """Load a YOLO model and store the confidence threshold for detections."""
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect_frame(self, frame):
        """Track persons and sports balls in a frame and return detection dictionaries."""
        results = self.model.track(frame, persist=True, conf=self.confidence, classes=[0, 32])
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                track_id = int(box.id[0]) if box.id is not None else None

                detections.append(
                    {
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": float(box.conf[0]),
                        "bbox": [x1, y1, x2, y2],
                        "center": [cx, cy],
                    }
                )

        return detections

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels for detections on a copy of the frame."""
        annotated = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = [int(v) for v in detection["bbox"]]
            color = (0, 255, 0) if detection["class_id"] == 0 else (0, 0, 255)
            label = f"{detection['class_name']} {detection['track_id']}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return annotated
