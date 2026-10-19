from pathlib import Path

import numpy as np

COCO_SPORTS_BALL = 32
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "ball_yolov8n.pt"


class BallDetector:
    """Finds the basketball in a frame with a YOLO model.

    Uses the fine-tuned weights in models/ball_yolov8n.pt when present, otherwise falls back to the
    off-the-shelf COCO YOLOv8n and its "sports ball" class.
    """

    def __init__(self, weights=None, conf=0.25):
        from ultralytics import YOLO
        weights = Path(weights) if weights else DEFAULT_WEIGHTS
        if weights.exists():
            self.model, self.classes = YOLO(str(weights)), None
        else:
            self.model, self.classes = YOLO("yolov8n.pt"), [COCO_SPORTS_BALL]
        self.conf = conf

    def detect(self, frame_bgr):
        """Most confident ball as (center_x, center_y, radius, confidence) in pixels, or None."""
        result = self.model.predict(frame_bgr, conf=self.conf, classes=self.classes, verbose=False)[0]
        if len(result.boxes) == 0:
            return None
        best = int(result.boxes.conf.argmax())
        x1, y1, x2, y2 = result.boxes.xyxy[best].cpu().numpy()
        return ((x1 + x2) / 2, (y1 + y2) / 2, float(max(x2 - x1, y2 - y1) / 2), float(result.boxes.conf[best]))
