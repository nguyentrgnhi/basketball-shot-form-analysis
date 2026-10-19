# Ball detector weights

Put a YOLOv8 ball detector here as `ball_yolov8n.pt` and `analyze_shot.py` will use it automatically.
Without it, the code falls back to the generic COCO YOLOv8n model and its "sports ball" class.

Train your own from a Roboflow YOLO export with `scripts/train_ball_detector.py` (see the main README).
