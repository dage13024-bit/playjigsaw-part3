#!/usr/bin/env python3
"""
把自动打标的 Part 3 画面整理成 YOLO 分类数据集（开发期工具）

  * 输入：datasets/p3_labels.csv
  * 输出：datasets/p3cls/{train,val}/<类别>/*.png + datasets/p3cls/split.json

两个关键点：
  1. **按题库原图划分 train/val**，同一张照片不会同时进训练和验证；
  2. **分层划分**：每个类别至少留 2 个验证样本（能留就留），
     避免上一版"airplane/cup/desktop 验证集为 0"的问题。

用法：
  .venv/bin/python tools/build_p3_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _h(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


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

    # ---- 按 (类别, 原图) 分组 ----
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in kept:
        groups[r["key"]][r["path"]].append(r)

    # ---- 分层划分：每类抽 15~20% 的原图进 val，且尽量保证 ≥2 个 val 样本 ----
    val_paths: set[str] = set()
    print(f"\n{'类别':<10} {'原图数':>6} {'val原图':>7} {'train样本':>9} {'val样本':>7}")
    for cls in sorted(groups):
        paths = groups[cls]
        keys = sorted(paths, key=lambda p: _h(cls + p))
        if len(keys) <= 1:
            print(f"{cls:<10} {len(keys):>6} {'0':>7} {len(kept):>9} {'0':>7}  ⚠ 只有 1 张原图，无法分层")
            continue
        # 目标：至少 2 张原图、3 个样本进 val；但不超过 25%，且至少留 1 张给 train
        n_val = 2 if len(keys) >= 6 else 1
        while n_val < len(keys) - 1 and sum(len(paths[p]) for p in keys[:n_val]) < 3:
            n_val += 1
        n_val = min(n_val, max(1, len(keys) // 4))
        n_val = min(n_val, len(keys) - 1)
        val_paths.update(keys[:n_val])
        n_train = sum(len(paths[p]) for p in keys[n_val:])
        n_val_s = sum(len(paths[p]) for p in keys[:n_val])
        print(f"{cls:<10} {len(keys):>6} {n_val:>7} {n_train:>9} {n_val_s:>7}")

    # ---- 复制文件 ----
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
        split = "val" if r["path"] in val_paths else "train"
        dst_dir = out / split / r["key"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / f"{r['game']}_{src.name}")
        n[split] += 1

    (out / "split.json").write_text(json.dumps(
        {"val_paths": sorted(val_paths)}, ensure_ascii=False, indent=2))

    print(f"\ntrain {n['train']} / val {n['val']}")
    classes = sorted({p.name for p in (out / "train").iterdir() if p.is_dir()})
    print(f"{len(classes)} 类：{classes}")
    missing = [c for c in classes if not (out / "val" / c).exists()]
    if missing:
        print(f"⚠ 这些类别没有 val 样本：{missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
