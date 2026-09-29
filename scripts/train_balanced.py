import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the stratified TLFO split")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/tlfo_balanced.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="tlfo_yolov8s_balanced")
    parser.add_argument("--patience", type=int, default=40)
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
        close_mosaic=20,
        cos_lr=True,
        mixup=0.1,
        copy_paste=0.1,
        cls=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        cache=False,
    )


if __name__ == "__main__":
    main()
