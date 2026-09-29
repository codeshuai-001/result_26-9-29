import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train AKCMamba_YOLO / YOLOv8 on the TLFO dataset")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/tlfo.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="tlfo_yolov8n")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    data = args.data if Path(args.data).is_absolute() else str(root / args.data)

    model = YOLO(args.model)
    model.info()
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(root / args.project) if not Path(args.project).is_absolute() else args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        pretrained=True,
        amp=True,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
        close_mosaic=10,
    )


if __name__ == "__main__":
    main()
