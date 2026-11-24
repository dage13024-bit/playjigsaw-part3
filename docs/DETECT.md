# 检测模型（两套）

> 这是**交付用的检测版模型**，和主流程 `play_p3.py` 用的分类模型分开。
> 主流程仍然用 `p3cls_piece`（分类）；这两套检测模型用于 Part 1 风格的“目标检测”，
> 以及画布碎片定位的探索。

## 一、任务 A：主体检测（obj）

**输入**：一块碎片，或参考图里的一个格子（约 320×240 / 97×72）
**输出**：主体框 + 15 类（dog / cat / car / …）

- 数据集：`datasets/p3det_obj/`（909 train / 102 val）
- 标签来源：`p3_labels.csv` + **题库原图自动估框**
  - 背景干净（图边中位色差小）→ 背景分割取最大前景块
  - 背景复杂 → 中心 85% 兜底
  - 实测：910 张来自背景分割，101 张来自兜底
- **这是弱标注**：类别是准的（来自题库自动打标），框不完美；适合“分类/定位演示”，
  不适合按 COCO mAP 去要求它。
- 模型：`models/p3det_obj/weights/best.pt`
- 训练命令：

```bash
.venv-yolo/bin/yolo detect train data=datasets/p3det_obj/data.yaml \
    model=yolo11s.pt epochs=40 imgsz=320 batch=16 device=mps patience=15
```

## 二、任务 B：画布碎片检测（piece）

**输入**：整张拼图画布（1039×698 左右）
**输出**：30 个碎片框（1 类：`piece`）

- 数据集：`datasets/p3det_piece/`（85 train / 25 val）
- 标签来源：**槽位公式**（不是人工标）——每块碎片的核心框 + 卡扣外扩 1.15×tab
- 数据增强：每张画布再切 4 个 65% 象限图（22 张 → 110 张）
- 用途：替代/校验槽位定位；未来做“初始重叠”开局时，碎片不再等距，检测比公式更合适
- 模型：`models/p3det_piece/weights/best.pt`
- 训练命令：

```bash
.venv-yolo/bin/yolo detect train data=datasets/p3det_piece/data.yaml \
    model=yolo11s.pt epochs=60 imgsz=640 batch=8 device=mps patience=20
```

## 三、怎么跑

```bash
# 主体检测：一张参考图
.venv-yolo/bin/python tools/demo_detect.py \
    --model models/p3det_obj/weights/best.pt \
    --source logs/demo_ref.png --out logs/det_obj_demo.png

# 碎片检测：一张画布
.venv-yolo/bin/python tools/demo_detect.py \
    --model models/p3det_piece/weights/best.pt \
    --source logs/demo_canvas.png --out logs/det_piece_demo.png
```

## 四、指标

| 模型 | 数据 | 指标 |
|---|---|---|
| obj | 909 / 102 | **mAP50 = 0.880，mAP50-95 = 0.776**（102 张 val，弱框） |
| piece | 85 / 25 | **mAP50 = 0.994，mAP50-95 = 0.803**（25 张 val，338 个框） |

## 五、限制

- obj 的框是弱标注，**不要拿去和人工精标的 mAP 比**；它的价值是“类别 + 大致位置”。
- piece 只有开局帧（碎片在托盘），没有“拼到一半”“初始重叠”的样本；
  在真实重叠局面上泛化能力未验证。
- 两套检测模型都**没有接入** `play_p3.py` 主流程；主流程的成功率仍由分类 + NCC 保证。
