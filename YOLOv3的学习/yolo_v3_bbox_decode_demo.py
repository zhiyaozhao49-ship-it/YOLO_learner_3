"""
YOLOv3 Bounding Box Decode（边界框解码）教学脚本
==================================================

功能：实现 YOLOv3 的边界框解码函数，将网络预测值 (tx, ty, tw, th)
      转换为图像上的实际坐标 (bx, by, bw, bh)。

      不训练、不完整推理，只演示解码的数学过程。

YOLOv3 边界框解码公式（论文中 Figure 2）:

    bx = σ(tx) + cx          中心 x = sigmoid(tx) + 网格列号
    by = σ(ty) + cy          中心 y = sigmoid(ty) + 网格行号
    bw = pw × e^(tw)         宽度   = anchor宽 × exp(tw)
    bh = ph × e^(th)         高度   = anchor高 × exp(th)

    其中:
        (cx, cy) = 当前网格的左上角坐标（整数，0 开始计数）
        (pw, ph) = 当前 anchor 的预设宽高
        σ        = sigmoid 函数

参考：YOLOv3 论文 —— "An Incremental Improvement" (2018)
"""

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════(tx, ty, tw, th) → (bx, by, bw, bh)
# ═══════════════════════════════════════════════════════════════════════════════

def decode_bbox(prediction: torch.Tensor,
                anchors: torch.Tensor,
                stride: int) -> torch.Tensor:
    """
    YOLOv3 边界框解码函数。

    参数:
        prediction: 网络预测张量, shape [B, grid_h, grid_w, num_anchors, 85]
                    最后一维 85 = [tx, ty, tw, th, conf, p0, p1, ..., p79]
        anchors:    当前尺度的 anchor 宽高, shape [num_anchors, 2]
                    例如 [[116, 90], [156, 198], [373, 326]]
                    注意: 这里是像素单位，函数内部会除以 stride 转 cell 单位
        stride:     当前尺度的下采样步长
                    13×13 → stride=32, 26×26 → stride=16, 52×52 → stride=8

    返回:
        decoded: 解码后的坐标张量, shape [B, grid_h, grid_w, num_anchors, 4]
                 最后一维 4 = [bx, by, bw, bh]，单位为 grid cell
                 后续乘 stride 可还原为像素坐标
    """

    # ── 1. 提取前 4 个值: tx, ty, tw, th ──────────────────────────────
    # prediction[..., 0:4] 取所有 anchor 的坐标预测
    tx = prediction[..., 0]   # 中心 x 的原始预测值
    ty = prediction[..., 1]   # 中心 y 的原始预测值
    tw = prediction[..., 2]   # 宽度缩放因子
    th = prediction[..., 3]   # 高度缩放因子

    # ── 2. 构建网格坐标 cx, cy ────────────────────────────────────────
    # cx, cy 是每个网格左上角在特征图上的整数坐标
    # 例如 13×13 特征图: cx 从 0 到 12，cy 从 0 到 12
    #
    # 为什么需要 cx, cy？
    #   sigmoid(tx) 输出 0~1 之间的小数，表示"在当前网格内的偏移"
    #   加上 cx/cy 后才是"在整个特征图上的绝对位置"

    grid_h = prediction.shape[1]
    grid_w = prediction.shape[2]

    # meshgrid 生成网格坐标矩阵
    # cy[i,j] = i (行号), cx[i,j] = j (列号)
    cy, cx = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )

    # 扩展维度以匹配 [B, grid_h, grid_w, num_anchors]
    # cx/cy 原本是 [grid_h, grid_w]，需要变成 [1, grid_h, grid_w, 1]
    # 以便和 [B, grid_h, grid_w, num_anchors] 做广播
    cx = cx[None, :, :, None]   # [1, H, W, 1]
    cy = cy[None, :, :, None]   # [1, H, W, 1]

    # ── 3. 解码中心坐标: bx = σ(tx) + cx,  by = σ(ty) + cy ─────────
    #
    # sigmoid(tx) 将网络输出压缩到 (0, 1)，含义是:
    #   "预测框的中心在当前网格内的水平偏移比例"
    #   0.0 = 网格左边缘,  1.0 = 网格右边缘
    #
    # 加上 cx（网格列号）后，bx 就是中心点在特征图上的绝对 x 坐标
    #
    # 数学: bx = σ(tx) + cx
    #       by = σ(ty) + cy
    #
    # 例: cx=5, tx→sigmoid=0.7 → bx = 5.7
    #     表示中心在第 5 列网格内偏右 70% 的位置

    bx = torch.sigmoid(tx) + cx
    by = torch.sigmoid(ty) + cy

    # ── 4. 解码宽高: bw = pw × e^(tw),  bh = ph × e^(th) ───────────
    #
    # exp(tw) 将网络输出映射到 (0, +∞)，含义是:
    #   "anchor 宽度需要被放大多少倍"
    #   tw=0 → exp(0)=1 → bw=pw（和 anchor 一样宽）
    #   tw=2 → exp(2)≈7.4 → bw 是 anchor 的 7.4 倍
    #   tw=-1 → exp(-1)≈0.37 → bw 是 anchor 的 0.37 倍
    #
    # 为什么用 exp 而不直接预测宽度？
    #   · 宽度必须 > 0，exp 保证输出恒正
    #   · 使用 log 空间预测使得大小框的梯度更均衡
    #   · tw=0 时 bw=pw，初始状态接近 anchor 先验，训练更稳定
    #
    # 注意: anchor 宽高需要除以 stride 转为 cell 单位
    #   例如 anchor 宽 116px, stride=32 → pw = 116/32 = 3.625 cells

    # anchors: [num_anchors, 2] → [1, 1, 1, num_anchors, 2]
    pw = anchors[:, 0][None, None, None, :] / stride   # cell 单位
    ph = anchors[:, 1][None, None, None, :] / stride

    bw = pw * torch.exp(tw)
    bh = ph * torch.exp(th)

    # ── 5. 合并结果 ───────────────────────────────────────────────────
    # stack 在最后一个维度上拼接 [bx, by, bw, bh]
    decoded = torch.stack([bx, by, bw, bh], dim=-1)

    return decoded


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序：用随机输入测试解码函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("  YOLOv3 Bounding Box Decode（边界框解码）演示")
    print("=" * 70)

    # ── 模拟输入 ───────────────────────────────────────────────────────
    # 以 13×13 尺度为例 (stride=32, 大目标)

    batch = 2
    grid_h, grid_w = 13, 13
    num_anchors = 3
    num_values = 85

    # 随机预测值（实际来自网络输出）
    prediction = torch.randn(batch, grid_h, grid_w, num_anchors, num_values)

    # 13×13 尺度的 3 个 anchor（像素单位，相对 416×416）
    anchors = torch.tensor([
        [116,  90],   # anchor 0: 大目标偏正方形
        [156, 198],   # anchor 1: 大目标偏高
        [373, 326],   # anchor 2: 超大目标
    ], dtype=torch.float32)

    stride = 32  # 13×13 的下采样步长

    print(f"\n  输入 prediction shape: {list(prediction.shape)}")
    print(f"  Anchor (像素): {anchors.tolist()}")
    print(f"  Stride: {stride}")
    print(f"  Anchor (cell): {[[w/stride, h/stride] for w, h in anchors.tolist()]}")

    # ── 解码 ───────────────────────────────────────────────────────────
    decoded = decode_bbox(prediction, anchors, stride)

    print(f"\n  解码输出 shape: {list(decoded.shape)}")
    print(f"  [batch={batch}, grid_h={grid_h}, grid_w={grid_w}, "
          f"anchors={num_anchors}, coords=4]")

    # ── 展示具体数值 ───────────────────────────────────────────────────
    print(f"\n  解码示例（batch=0, 网格位置 (0,0) 的 3 个 anchor）:")
    print(f"  {'anchor':>8s}  {'bx':>8s}  {'by':>8s}  {'bw':>8s}  {'bh':>8s}  "
          f"{'说明':>20s}")
    print("  " + "-" * 70)

    for a in range(num_anchors):
        bx = decoded[0, 0, 0, a, 0].item()
        by = decoded[0, 0, 0, a, 1].item()
        bw = decoded[0, 0, 0, a, 2].item()
        bh = decoded[0, 0, 0, a, 3].item()
        note = f"anchor({int(anchors[a,0])},{int(anchors[a,1])})"
        print(f"  {a:>8d}  {bx:>8.3f}  {by:>8.3f}  {bw:>8.3f}  {bh:>8.3f}  "
              f"{note:>20s}")

    # ── 验证 sigmoid 约束 ──────────────────────────────────────────────
    # bx 的范围应该接近 [cx, cx+1]，因为 sigmoid(tx) ∈ (0, 1)
    print(f"\n  验证: sigmoid 将中心坐标约束在网格内部")
    print(f"  bx 范围: [{decoded[..., 0].min():.3f}, "
          f"{decoded[..., 0].max():.3f}]")
    print(f"  by 范围: [{decoded[..., 1].min():.3f}, "
          f"{decoded[..., 1].max():.3f}]")
    print(f"  (bx 在 [0, grid_w] 之间，by 在 [0, grid_h] 之间)")

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  解码公式详解")
    print("=" * 70)
    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │  bx = σ(tx) + cx        中心 x = sigmoid(tx) + 网格列号     │
  │  by = σ(ty) + cy        中心 y = sigmoid(ty) + 网格行号     │
  │  bw = pw × e^(tw)       宽度 = anchor宽 × exp(tw)          │
  │  bh = ph × e^(th)       高度 = anchor高 × exp(th)          │
  └──────────────────────────────────────────────────────────────┘

  1. 中心坐标 (bx, by):
     · sigmoid(tx) 输出 (0, 1)，代表中心在网格内的相对偏移
     · + cx/cy 把相对偏移转为特征图上的绝对坐标
     · sigmoid 保证了预测框中心不会"跑出"当前网格（防止框乱飘）

  2. 宽高 (bw, bh):
     · exp(tw) 输出 (0, +∞)，代表 anchor 的缩放倍数
     · × pw/ph 把缩放倍数转为实际宽高（cell 单位）
     · exp 保证了宽高恒正（框不可能有负宽度）

  3. 转像素坐标（最终画框时）:
     · pixel_x = bx × stride
     · pixel_y = by × stride
     · pixel_w = bw × stride
     · pixel_h = bh × stride
     · 然后转换为左上角坐标: x1 = pixel_x - pixel_w/2
                            y1 = pixel_y - pixel_h/2
                            x2 = pixel_x + pixel_w/2
                            y2 = pixel_y + pixel_h/2
    """)


if __name__ == "__main__":
    main()
