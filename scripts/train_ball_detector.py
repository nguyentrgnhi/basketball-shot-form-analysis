"""Fine-tune a small YOLOv8 model to detect a basketball.

    python scripts/train_ball_detector.py --dataset /path/to/Basketball-Tracking-2

The dataset is a Roboflow export in YOLO format (train/valid/test folders with images/ and labels/).
It is not part of this repository. The best weights are copied to models/ball_yolov8n.pt.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="folder with train/ valid/ test/ (YOLO format)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2, help="data loader workers (lower = less memory)")
    parser.add_argument("--device", default=None, help="e.g. cpu, mps, 0 (default: let Ultralytics choose)")
    parser.add_argument("--output", default=str(ROOT / "models" / "ball_yolov8n.pt"))
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    # Roboflow's data.yaml uses paths relative to itself that break when the folder moves; write our own.
    config = {"path": str(dataset), "train": "train/images", "val": "valid/images", "test": "test/images",
              "names": {0: "basketball"}}
    with tempfile.TemporaryDirectory() as tmp:
        data_yaml = Path(tmp) / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(config))
        model = YOLO("yolov8n.pt")  # COCO-pretrained starting point
        model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, workers=args.workers, patience=15, seed=0, project=str(ROOT / "runs"), name="ball", exist_ok=True)
        best = Path(model.trainer.best)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
