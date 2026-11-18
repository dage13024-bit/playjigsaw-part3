#!/usr/bin/env python3
"""
把 datasets/catalog 整理成 YOLO 分类格式（train/val 分层切分）。

  datasets/cls/train/<key>/*.jpg
  datasets/cls/val/<key>/*.jpg

用法：.venv/bin/python tools/build_cls_dataset.py [--val-ratio 0.15]
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "catalog"
DST = ROOT / "datasets" / "cls"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SRC / "labels.csv")))
    by_cls: dict[str, list[str]] = {}
    for r in rows:
        by_cls.setdefault(r["key"], []).append(r["path"])

    rng = random.Random(args.seed)
    if DST.exists():
        shutil.rmtree(DST)
    n_tr = n_va = 0
    for key, paths in sorted(by_cls.items()):
        paths = sorted(paths)
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * args.val_ratio)))
        n_val = min(n_val, max(1, len(paths) - 1))
        val, train = paths[:n_val], paths[n_val:]
        for split, items in (("train", train), ("val", val)):
            d = DST / split / key
            d.mkdir(parents=True, exist_ok=True)
            for rel in items:
                shutil.copy2(SRC / rel, d / Path(rel).name)
        n_tr += len(train); n_va += len(val)
        print(f"  {key:10s} train {len(train):3d}  val {len(val):2d}")
    print(f"\n合计 train {n_tr} / val {n_va} -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
