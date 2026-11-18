#!/usr/bin/env python3
"""
训练 YOLO11s 分类器（15 类题库）。

  .venv-yolo/bin/python tools/train_cls.py --epochs 40

产物：models/cls_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "cls"
import os
if os.environ.get("DATA_DIR"):
    DATA = ROOT / os.environ["DATA_DIR"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="cls_v1")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    model = YOLO("yolo11s-cls.pt")
    model.train(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device,
                project=str(ROOT / "models"), name=args.name, exist_ok=True,
                pretrained=True, val=True, plots=False, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
