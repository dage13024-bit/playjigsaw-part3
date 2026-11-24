#!/usr/bin/env python3
"""
构建两套 YOLO 检测数据集（开发期工具）

任务 A（obj）：输入一块碎片 / 一个参考格，检测里面的主体（框 + 15 类）
    - 标签框从**题库原图**自动估：背景分割取最大前景块，失败则用中心 85% 兜底
    - 这是"弱标注"：类别准，框不完美；用于交付一版检测模型

任务 B（piece）：输入整张画布，检测 30 块碎片（1 类）
    - 标签从**槽位公式**直接得到（x, y, pw, ph），精确
    - 只有开局帧（碎片都在托盘），22 局 ≈ 22 张图

用法：
  .venv/bin/python tools/build_det_dataset.py --task obj
  .venv/bin/python tools/build_det_dataset.py --task piece
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def auto_box(img: np.ndarray):
    """题库原图上的主体框：背景分割 + 兜底。"""
    h, w = img.shape[:2]
    k = max(2, int(0.02 * min(h, w)))
    border = np.concatenate([
        img[:k].reshape(-1, 3), img[-k:].reshape(-1, 3),
        img[:, :k].reshape(-1, 3), img[:, -k:].reshape(-1, 3)])
    med = np.median(border, axis=0)
    mad = float(np.median(np.abs(border - med)))
    if mad < 25:                                   # 背景干净（白底/纯色底）
        d = np.linalg.norm(img.astype(np.float32) - med, axis=2)
        m = (d > 40).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            x, y, ww, hh, area = (int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
                                  int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT]),
                                  int(st[i, cv2.CC_STAT_AREA]))
            frac = area / float(w * h)
            if 0.03 < frac < 0.95 and ww > 0.1 * w and hh > 0.1 * h:
                return x, y, ww, hh, "bg"
    mx, my = int(w * 0.05), int(h * 0.05)
    return mx, my, w - 2 * mx, h - 2 * my, "center"


def write_label(root: Path, split: str, name: str, boxes, names: list[str],
                iw: int, ih: int):
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, (x, y, w, h) in boxes:
        x = max(0.0, min(float(x), iw - 1)); y = max(0.0, min(float(y), ih - 1))
        w = max(1.0, min(float(w), iw - x)); h = max(1.0, min(float(h), ih - y))
        lines.append(f"{names.index(cls)} {(x + w / 2) / iw:.6f} "
                     f"{(y + h / 2) / ih:.6f} {w / iw:.6f} {h / ih:.6f}")
    (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines))


def write_sample(root: Path, split: str, name: str, img: np.ndarray,
                 boxes, names: list[str]):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "images" / split / f"{name}.png"), img)
    ih, iw = img.shape[:2]
    write_label(root, split, name, boxes, names, iw, ih)


def build_obj(args) -> None:
    rows = [r for r in csv.DictReader(open(ROOT / "datasets" / "p3_labels.csv"))
            if float(r["score"]) >= args.min_score]
    split = json.loads((ROOT / "datasets" / "p3cls" / "split.json").read_text())["val_paths"]
    val_paths = set(split)
    classes = sorted({r["key"] for r in rows})
    root = ROOT / "datasets" / "p3det_obj"
    if root.exists():
        shutil.rmtree(root)

    stats = {"train": 0, "val": 0, "bg": 0, "center": 0}
    for r in rows:
        gdir = ROOT / "datasets" / "p3_raw" / r["game"]
        src = (gdir / r["file"].replace("_piece", "_core")) if r["kind"] == "piece" else (gdir / r["file"])
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        cat = cv2.imread(str(ROOT / "datasets" / "catalog" / r["path"]))
        if img is None or cat is None:
            continue
        x, y, w, h, how = auto_box(cat)
        ih, iw = img.shape[:2]
        sx, sy = iw / cat.shape[1], ih / cat.shape[0]
        box = (x * sx, y * sy, w * sx, h * sy)
        sp = "val" if r["path"] in val_paths else "train"
        name = f"{r['game']}_{src.stem}"
        (root / "images" / sp).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, root / "images" / sp / f"{name}.png")
        write_label(root, sp, name, [(r["key"], box)], classes, iw, ih)
        stats[sp] += 1
        stats[how] += 1

    yaml = ["path: " + str(root), "train: images/train", "val: images/val",
            "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(classes)]
    (root / "data.yaml").write_text("\n".join(yaml) + "\n")
    print(f"obj: train {stats['train']} / val {stats['val']}  "
          f"框来源 bg {stats['bg']} / center {stats['center']}")


def build_piece(args) -> None:
    from pipeline import detect_board
    from slots import occupied_slots

    root = ROOT / "datasets" / "p3det_piece"
    if root.exists():
        shutil.rmtree(root)
    games = sorted((ROOT / "datasets" / "p3_raw").glob("g*/g_canvas.png"))
    stats = {"train": 0, "val": 0, "boxes": 0}
    for i, p in enumerate(games):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        board = detect_board(img)
        if board is None:
            continue
        s = 5
        pw, ph = board["w"] / s, board["h"] / s
        tab = 0.22 * min(pw, ph)
        occ = occupied_slots(img, board, pw, ph, expect=30)
        boxes = []
        for slot in occ:
            # 框住整个碎片（含伸出的卡扣）。展示时可用 --snap 吸附回核心框。
            x = slot["x"] - tab * 1.15
            y = slot["y"] - tab * 1.15
            boxes.append(("piece", (x, y, pw + 2 * tab * 1.15, ph + 2 * tab * 1.15)))
        sp = "val" if i % 5 == 0 else "train"
        write_sample(root, sp, p.parent.name, img[:, :, :3], boxes, ["piece"])
        stats[sp] += 1
        stats["boxes"] += len(boxes)

        # 数据增强：再切 4 个 65% 的象限图，22 张 → 110 张，缓解小样本过拟合
        h, w = img.shape[:2]
        tw, th = int(w * 0.65), int(h * 0.65)
        for t, (x0, y0) in enumerate([(0, 0), (w - tw, 0), (0, h - th), (w - tw, h - th)]):
            crop = img[y0:y0 + th, x0:x0 + tw, :3]
            keep = []
            for cls, (x, y, ww, hh) in boxes:
                cx, cy = x + ww / 2, y + hh / 2
                if x0 <= cx < x0 + tw and y0 <= cy < y0 + th:
                    keep.append((cls, (x - x0, y - y0, ww, hh)))
            if len(keep) >= 3:
                write_sample(root, sp, f"{p.parent.name}_t{t}", crop, keep, ["piece"])
                stats[sp] += 1
                stats["boxes"] += len(keep)

    (root / "data.yaml").write_text(
        "path: " + str(root) + "\ntrain: images/train\nval: images/val\n"
        "names:\n  0: piece\n")
    print(f"piece: train {stats['train']} / val {stats['val']}  "
          f"共 {stats['boxes']} 个框")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["obj", "piece"], required=True)
    ap.add_argument("--min-score", type=float, default=0.55)
    args = ap.parse_args()
    if args.task == "obj":
        build_obj(args)
    else:
        build_piece(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
