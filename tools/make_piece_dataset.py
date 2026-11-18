#!/usr/bin/env python3
"""
从题库图片生成"合成碎片图"，让训练分布贴近真实碎片。

为什么需要：题库图是整张照片，而考核里喂给分类器的是
"被拼图裁过、只占格子一部分、边缘还有凹槽"的小图 —— 分布差太远，
直接用整图训练，实测会在碎片上把 cat/cup/nailong 认混。

做法：把题库图随机裁一块 → 套上真实的拼图卡扣蒙版（复用 white_vision 的
Bézier 模板）→ 非碎片区域填黑（跟运行时裁出来的图一致）→ 存成分类数据集。

  .venv/bin/python tools/make_piece_dataset.py --per-image 5
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import white_vision as wv          # noqa: E402

SRC = ROOT / "datasets" / "catalog"
DST = ROOT / "datasets" / "piece"
PW, PH = 160.0, 120.0             # 合成用的画布尺寸（会缩放到 224 训练）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-image", type=int, default=5)
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tpls, (W, H), tab = wv._build_templates(PW, PH)
    rows = list(csv.DictReader(open(SRC / "labels.csv")))
    by_cls: dict[str, list[dict]] = {}
    for r in rows:
        by_cls.setdefault(r["key"], []).append(r)

    n_tr = n_va = 0
    for key, items in sorted(by_cls.items()):
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * args.val_ratio)))
        val_keys = {id(x) for x in items[:n_val]}
        for it in items:
            src = cv2.imread(str(SRC / it["path"]))
            if src is None:
                continue
            split = "val" if id(it) in val_keys else "train"
            d = DST / split / key
            d.mkdir(parents=True, exist_ok=True)
            for v in range(args.per_image):
                # 随机裁一块：模拟"目标只占格子一部分"
                f = rng.uniform(0.62, 0.95)
                h, w = src.shape[:2]
                cw, chh = max(8, int(w * f)), max(8, int(h * f))
                x0 = rng.randint(0, w - cw); y0 = rng.randint(0, h - chh)
                crop = src[y0:y0 + chh, x0:x0 + cw]
                crop = cv2.resize(crop, (W, H))
                t = rng.choice(tpls)
                m = t["mask"]
                bg = crop.mean(axis=(0, 1))                  # 外侧填图像均值色，避免"黑边"偏置
                out = np.tile(bg.astype(np.uint8), (H, W, 1))
                sel = m[:H, :W] > 0
                out[sel] = crop[sel]
                tb = t["bbox"]
                piece = out[tb[1]:tb[1] + tb[3], tb[0]:tb[0] + tb[2]]
                cv2.imwrite(str(d / f"{key}-{it['path'].split('/')[-1][:-4]}-v{v}.jpg"),
                            piece, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if split == "train":
                    n_tr += 1
                else:
                    n_va += 1
        print(f"  {key:10s} -> train/val 已生成")
    print(f"\n合计 train {n_tr} / val {n_va} -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
