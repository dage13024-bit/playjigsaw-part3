#!/usr/bin/env python3
"""
检测模型演示：对一张图（或一个目录）跑 YOLO 检测，把框画出来。

用法：
    .venv-yolo/bin/python tools/demo_detect.py \
        --model models/p3det_obj/weights/best.pt \
        --source logs/demo_ref.png \
        --out logs/det_demo.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="logs/det_demo.png")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.model)
    src = Path(args.source)
    files = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for f in files:
        r = model.predict(str(f), conf=args.conf, verbose=False)[0]
        img = r.plot()
        dst = out if len(files) == 1 else out.with_name(f"{out.stem}_{f.stem}{out.suffix}")
        cv2.imwrite(str(dst), img)
        boxes = r.boxes
        print(f"{f.name}: {len(boxes)} 个框")
        for b in boxes[:10]:
            cls = r.names[int(b.cls[0])]
            print(f"  {cls:<10} conf={float(b.conf[0]):.2f} "
                  f"xyxy={[round(v, 1) for v in b.xyxy[0].tolist()]}")
        print(f"  → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
