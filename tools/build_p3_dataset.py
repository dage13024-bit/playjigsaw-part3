#!/usr/bin/env python3
"""
把自动打标的 Part 3 画面整理成 YOLO 分类数据集（开发期工具）

  * 输入：datasets/p3_labels.csv（tools/label_p3.py 产出）
  * 输出：datasets/p3cls/{train,val}/<类别>/*.png

两个关键点：
  1. **按题库原图划分 train/val**，不按对局划分 —— 否则同一张照片会同时出现在
     训练集和验证集，val 准确率虚高。
  2. 碎片用"完整核心 + 灰底合成"那张（和推理时完全同一个域），
     格子用参考图原生切格。

用法：
  .venv/bin/python tools/build_p3_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def is_val(path: str, every: int = 7) -> bool:
    h = hashlib.md5(path.encode()).hexdigest()
    return int(h[:8], 16) % every == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "datasets" / "p3_labels.csv"))
    ap.add_argument("--raw", default=str(ROOT / "datasets" / "p3_raw"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "p3cls"))
    ap.add_argument("--min-score", type=float, default=0.55)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.labels)))
    kept = [r for r in rows if float(r["score"]) >= args.min_score]
    print(f"标签 {len(rows)} 条，保留 {len(kept)} 条（score≥{args.min_score}）")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / split).mkdir(parents=True)

    n = {"train": 0, "val": 0}
    for r in kept:
        gdir = Path(args.raw) / r["game"]
        if r["kind"] == "piece":
            src = gdir / r["file"].replace("_piece", "_core")
            if not src.exists():
                src = gdir / r["file"]
        else:
            src = gdir / r["file"]
        if not src.exists():
            continue
        split = "val" if is_val(r["path"]) else "train"
        dst_dir = out / split / r["key"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / f"{r['game']}_{src.name}")
        n[split] += 1

    print(f"train {n['train']} / val {n['val']}")
    classes = sorted({p.name for p in (out / "train").iterdir() if p.is_dir()})
    print(f"{len(classes)} 类：{classes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
