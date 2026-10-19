"""Compare ball detectors on a held-out test split with the same metric.

    python scripts/evaluate_ball_detector.py --dataset /path/to/Basketball-Tracking-2

Scores (1) the off-the-shelf COCO YOLOv8n, using its "sports ball" class, and (2) the fine-tuned
model in models/ball_yolov8n.pt. AP is the area under the precision-recall curve (all-point
interpolation); AP50 needs IoU >= 0.5 and AP50-95 averages IoU thresholds 0.5, 0.55, ..., 0.95.
"""
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
COCO_SPORTS_BALL = 32
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_ground_truth(images_dir: Path):
    """{image path: array of [x1, y1, x2, y2] in pixels} from YOLO-format label files."""
    from PIL import Image
    truth = {}
    for image in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        w, h = Image.open(image).size
        label = images_dir.parent / "labels" / f"{image.stem}.txt"
        boxes = []
        if label.exists():
            for line in label.read_text().splitlines():
                if line.strip():
                    _, cx, cy, bw, bh = map(float, line.split()[:5])
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        truth[image] = np.array(boxes).reshape(-1, 4)
    return truth


def iou(box, boxes):
    x1, y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x2, y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def average_precision(detections, truth, iou_threshold):
    """detections: [(image, confidence, box)]. Greedy matching by confidence, one detection per ground-truth box."""
    n_truth = sum(len(b) for b in truth.values())
    matched = {image: np.zeros(len(boxes), bool) for image, boxes in truth.items()}
    tp = []
    for image, _, box in sorted(detections, key=lambda d: -d[1]):
        gt = truth[image]
        hit = False
        if len(gt):
            overlaps = iou(box, gt)
            overlaps[matched[image]] = 0
            best = int(overlaps.argmax())
            if overlaps[best] >= iou_threshold:
                matched[image][best] = True
                hit = True
        tp.append(hit)
    tp = np.array(tp, float)
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = cum_tp / max(n_truth, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    # all-point interpolated area under the precision-recall curve
    mrec, mpre = np.concatenate([[0], recall, [1]]), np.concatenate([[1], precision, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), recall, precision


def detect(model, images, only_class=None):
    detections = []
    for image in images:
        result = model.predict(str(image), conf=0.001, verbose=False, classes=None if only_class is None else [only_class])[0]
        for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            detections.append((image, float(conf), box))
    return detections


def score(detections, truth):
    ap50, recall, precision = average_precision(detections, truth, 0.5)
    ap = np.mean([average_precision(detections, truth, t)[0] for t in np.arange(0.5, 0.96, 0.05)])
    # operating point at confidence >= 0.25 (the usual default threshold)
    kept = [d for d in detections if d[1] >= 0.25]
    tp = sum(1 for d in kept if len(truth[d[0]]) and iou(d[2], truth[d[0]]).max() >= 0.5)
    n_truth = sum(len(b) for b in truth.values())
    return {"AP50": ap50, "AP50-95": float(ap), "precision@0.25": tp / max(len(kept), 1), "recall@0.25": tp / max(n_truth, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--weights", default=str(ROOT / "models" / "ball_yolov8n.pt"))
    args = parser.parse_args()

    images_dir = Path(args.dataset) / args.split / "images"
    truth = load_ground_truth(images_dir)
    print(f"{len(truth)} images, {sum(len(b) for b in truth.values())} ball boxes in the {args.split} split\n")

    rows = {
        "COCO YOLOv8n ('sports ball' class)": score(detect(YOLO("yolov8n.pt"), truth, only_class=COCO_SPORTS_BALL), truth),
        "Fine-tuned YOLOv8n": score(detect(YOLO(args.weights), truth), truth),
    }
    header = f"{'model':<38}{'AP50':>8}{'AP50-95':>10}{'prec@.25':>10}{'rec@.25':>9}"
    print(header + "\n" + "-" * len(header))
    for name, m in rows.items():
        print(f"{name:<38}{m['AP50']:>8.3f}{m['AP50-95']:>10.3f}{m['precision@0.25']:>10.3f}{m['recall@0.25']:>9.3f}")


if __name__ == "__main__":
    main()
