"""
YOLOv3 三尺度输出形状模拟脚本
==============================

功能：模拟 YOLOv3 检测头的输出张量形状，解释为什么每个尺度输出 255 通道。
      不包含骨干网络、不加载数据集、不写训练循环。

核心公式：
    输出通道数 = anchors_per_scale × (5 + classes)
              = 3 × (4 + 1 + 80)
              = 255

    其中：
        4  = 边界框坐标偏移量 (tx, ty, tw, th)
        1  = 目标存在置信度 (objectness)
        80 = COCO 数据集类别数 (class probabilities)

参考：YOLOv3 论文 —— "An Incremental Improvement" (2018)
"""

import torch


# ═══════════════════════════════════════════════════════════════════════════════
# 配置参数
# ═══════════════════════════════════════════════════════════════════════════════

BATCH_SIZE = 1        # 批次大小
INPUT_SIZE = 416      # 输入图片边长（像素）
CLASSES = 80          # COCO 数据集类别数
ANCHORS_PER_SCALE = 3 # 每个尺度的 anchor 数量

# 每个预测框输出的分量数
BBOX_COORDS = 4       # 边界框: tx, ty, tw, th
OBJECTNESS = 1        # 目标置信度: Po（该位置是否有物体）
# 通道数 = anchors × (bbox_coords + objectness + classes)
CHANNELS = ANCHORS_PER_SCALE * (BBOX_COORDS + OBJECTNESS + CLASSES)


# ═══════════════════════════════════════════════════════════════════════════════
# 三个尺度的空间尺寸
# ═══════════════════════════════════════════════════════════════════════════════

SCALES = [
    {"name": "尺度 3（大目标）",  "size": 13, "stride": 32},
    {"name": "尺度 2（中目标）",  "size": 26, "stride": 16},
    {"name": "尺度 1（小目标）",  "size": 52, "stride":  8},
]


def main():
    print("=" * 70)
    print("  YOLOv3 三尺度输出形状模拟")
    print(f"  输入图片: {INPUT_SIZE}×{INPUT_SIZE}   类别数: {CLASSES}   "
          f"每尺度 anchor: {ANCHORS_PER_SCALE}")
    print("=" * 70)

    # ── 1. 255 是怎么算出来的？逐步拆解 ────────────────────────────────
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  255 = {ANCHORS_PER_SCALE} × ({BBOX_COORDS} + {OBJECTNESS} + {CLASSES})                           │
  │      = anchors × (bbox + objectness + classes)           │
  └─────────────────────────────────────────────────────────┘

  拆解:
    · {BBOX_COORDS} 个边界框坐标 (tx, ty, tw, th)
        tx, ty: 中心点偏移（相对于当前网格左上角）
        tw, th: 宽高缩放因子（相对于 anchor 的宽高）

    · {OBJECTNESS} 个目标置信度 (objectness score)
        表示"这个 anchor 位置是否存在物体"
        训练时: 有物体=1, 无物体=0
        推理时: 输出 0~1 的概率值，用于过滤低置信度预测

    · {CLASSES} 个类别概率 (class predictions)
        COCO 数据集有 80 类: 人、车、猫、狗...
        每个类别独立输出一个概率值（不用 softmax，用 sigmoid）
        原因: 一个 anchor 可能同时包含多个类别标签（如"人"和"自行车"）

    · {ANCHORS_PER_SCALE} 个 anchor（每个尺度）
        每个 anchor 独立做上述 ({BBOX_COORDS}+{OBJECTNESS}+{CLASSES}) 个预测
        所以通道数 = {ANCHORS_PER_SCALE} × {BBOX_COORDS + OBJECTNESS + CLASSES} = {CHANNELS}
  """)

    # ── 2. 模拟三个尺度的输出张量 ───────────────────────────────────────
    print("  三个尺度的输出 Tensor:")
    print("  " + "-" * 62)

    total_predictions = 0

    for scale in SCALES:
        s = scale["size"]

        # 模拟检测头输出: [batch, 255, H, W]
        # 实际 YOLOv3 中，这是骨干特征图经过几层卷积后的最终输出
        output = torch.randn(BATCH_SIZE, CHANNELS, s, s)

        # 这个尺度产生的预测框总数
        num_preds = s * s * ANCHORS_PER_SCALE
        total_predictions += num_preds

        print(f"  {scale['name']}")
        print(f"    特征图: {list(output.shape)}")
        print(f"    stride={scale['stride']}, "
              f"网格={s}×{s}, "
              f"anchor={ANCHORS_PER_SCALE}")
        print(f"    预测框数: {s}×{s}×{ANCHORS_PER_SCALE} = {num_preds:,}")
        print()

    # ── 3. 汇总 ──────────────────────────────────────────────────────────
    print("  " + "-" * 62)
    print(f"  总预测框数: {total_predictions:,}  "
          f"(对比: YOLOv2 = {13*13*5:,}, YOLOv1 = {7*7*2:,})")
    print("=" * 70)

    # ── 4. 通道维度含义的图示 ────────────────────────────────────────────
    print("""
  单个网格位置的 255 通道含义（以尺度 13×13 为例）:

    通道索引      内容                   数量
    ─────────────────────────────────────────
    0  ~  4      Anchor 1 的预测          5  (tx,ty,tw,th,conf)
    5  ~  9      Anchor 2 的预测          5  (tx,ty,tw,th,conf)
    10 ~ 14      Anchor 3 的预测          5  (tx,ty,tw,th,conf)
    15 ~ 94      Anchor 1 的 80 类概率   80
    95 ~ 174     Anchor 2 的 80 类概率   80
    175 ~ 254    Anchor 3 的 80 类概率   80
    ─────────────────────────────────────────
    合计: 3 × (5 + 80) = 255

  注意: 上面是一种常见理解方式。实际上 YOLOv3 原始实现中
  通道排列是逐 anchor 拼接的:
    [anchor1: 85 | anchor2: 85 | anchor3: 85]
  每个 anchor 的 85 = [tx, ty, tw, th, conf, p0, p1, ..., p79]
""")


if __name__ == "__main__":
    main()
