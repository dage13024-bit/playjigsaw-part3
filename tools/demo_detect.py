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
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="logs/det_demo.png")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--snap", action="store_true",
                    help="碎片检测专用：用槽位几何把预测框吸附成精确的核心框")
    ap.add_argument("--grid", type=int, default=5)
    args = ap.parse_args()

    model = YOLO(args.model)
    src = Path(args.source)
    files = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    snapper = None
    if args.snap:
        from pipeline import detect_board
        from slots import occupied_slots

        def snapper(path):
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.shape[2] < 4:
                return None
            b = detect_board(raw)
            if b is None:
                return None
            pw, ph = b["w"] / args.grid, b["h"] / args.grid
            slots = occupied_slots(raw, b, pw, ph, expect=30)
            canvas = raw[:, :, :3].copy()
            for sl in slots:
                cv2.rectangle(canvas,
                              (int(sl["x"]), int(sl["y"])),
                              (int(sl["x"] + pw), int(sl["y"] + ph)), (0, 200, 0), 2)
            return canvas

    for f in files:
        r = model.predict(str(f), conf=args.conf, verbose=False)[0]
        img = r.plot()
        if snapper is not None:
            snapped = snapper(f)
            if snapped is not None:
                img = snapped
        dst = out if len(files) == 1 else out.with_name(f"{out.stem}_{f.stem}{out.suffix}")
        cv2.imwrite(str(dst), img)
        boxes = r.boxes
        print(f"{f.name}: {len(boxes)} 个框" + ("（已吸附到槽位）" if args.snap else ""))
        for b in boxes[:10]:
            cls = r.names[int(b.cls[0])]
            print(f"  {cls:<10} conf={float(b.conf[0]):.2f} "
                  f"xyxy={[round(v, 1) for v in b.xyxy[0].tolist()]}")
        print(f"  → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
