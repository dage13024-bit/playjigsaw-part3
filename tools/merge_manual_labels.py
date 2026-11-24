#!/usr/bin/env python3
"""
把人工修正过的框合并进主体检测数据集，生成一个"人工验证集"版本。

  * 原数据集：datasets/p3det_obj（自动弱框）
  * 人工标注：datasets/p3det_obj/labels_manual/val/*.txt（tools/labeler 产出）
  * 输出：datasets/p3det_obj_manual
      - 有人工标注的图：80% 进 train，20% 进 val（val 只放人工标注，指标可信）
      - 没有人工标注的图：保留自动弱框，只进 train

用法：
  .venv/bin/python tools/merge_manual_labels.py
  .venv-yolo/bin/yolo detect train data=datasets/p3det_obj_manual/data.yaml \
      model=models/p3det_obj/weights/best.pt epochs=30 imgsz=320 batch=16 device=mps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets/p3det_obj"))
    ap.add_argument("--manual", default=str(ROOT / "datasets/p3det_obj/labels_manual/val"))
    ap.add_argument("--out", default=str(ROOT / "datasets/p3det_obj_manual"))
    ap.add_argument("--manual-only", action="store_true",
                    help="只用人工标注的图（80/20），不混入弱框")
    args = ap.parse_args()

    src, manual, out = Path(args.src), Path(args.manual), Path(args.out)
    manual_files = {p.stem: p for p in manual.glob("*.txt")}
    print(f"人工标注 {len(manual_files)} 张")

    if out.exists():
        shutil.rmtree(out)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True)
        (out / "labels" / sp).mkdir(parents=True)

    names = []
    with open(src / "data.yaml") as f:
        for line in f:
            line = line.strip()
            if line.startswith("names:"):
                continue
            if ":" in line and line.split(":")[0].strip().isdigit():
                names.append((int(line.split(":")[0]), line.split(":", 1)[1].strip()))

    n = {"train": 0, "val": 0, "manual": 0}
    if args.manual_only:
        index = {p.stem: p for sp in ("train", "val")
                 for p in (src / "images" / sp).glob("*.png")}
        for stem, lab in sorted(manual_files.items()):
            img = index.get(stem)
            if img is None:
                continue
            sp = "val" if int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
            shutil.copy2(img, out / "images" / sp / img.name)
            shutil.copy2(lab, out / "labels" / sp / f"{stem}.txt")
            n[sp] += 1
            n["manual"] += 1
        yaml = ["path: " + str(out), "train: images/train", "val: images/val", "names:"]
        yaml += [f"  {i}: {nm}" for i, nm in names]
        (out / "data.yaml").write_text("\n".join(yaml) + "\n")
        print(f"[manual-only] train {n['train']} / val {n['val']}")
        return 0

    for split in ("train", "val"):
        for img in sorted((src / "images" / split).glob("*.png")):
            stem = img.stem
            lab = src / "labels" / split / f"{stem}.txt"
            is_manual = stem in manual_files
            if is_manual:
                lab = manual_files[stem]
                n["manual"] += 1
                # 80/20 划分：按文件名哈希，保证同一图不会两边都有
                sp = "val" if int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
            else:
                sp = "train"
            shutil.copy2(img, out / "images" / sp / img.name)
            if lab.exists():
                shutil.copy2(lab, out / "labels" / sp / f"{stem}.txt")
            n[sp] += 1

    yaml = ["path: " + str(out), "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {nm}" for i, nm in names]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    print(f"train {n['train']}（含人工 {n['manual']}）/ val {n['val']}")
    print("注意：val 只含人工标注的图，指标比之前可信；train 里仍混有弱框，适合微调。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
