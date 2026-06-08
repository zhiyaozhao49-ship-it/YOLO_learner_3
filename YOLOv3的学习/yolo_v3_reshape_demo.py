"""
YOLOv3 输出张量 Reshape 实验脚本
=================================

功能：演示 YOLOv3 检测头输出如何从 [B, 255, H, W] 重排为
      [B, H, W, 3, 85]，为后续解码做准备。

      不训练、不推理，纯张量维度操作演示。

为什么要 reshape？
    网络原始输出的 255 通道是"拍平"的，所有 anchor 的预测混在一起：
        [anchor1: 85 | anchor2: 85 | anchor3: 85] → 255 通道

    但解码时我们需要"逐 anchor"访问每个预测值:
        tx, ty, tw, th, conf, p0, p1, ..., p79

    所以必须把 255 拆成 3 × 85，让每个 anchor 的 85 个值独立可索引。

参考：YOLOv3 论文 —— "An Incremental Improvement" (2018)
"""

import torch


# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

BATCH = 2          # 批次大小
CHANNELS = 255     # 3 anchors × (4 + 1 + 80)
GRID_H = 13        # 特征图高度
GRID_W = 13        # 特征图宽度
ANCHORS = 3        # 每个尺度的 anchor 数量
VALUES = 85        # 每个 anchor 的预测值数量: 4 + 1 + 80


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("  YOLOv3 输出张量 Reshape 实验")
    print("=" * 70)

    # ── Step 1: 模拟网络原始输出 [B, 255, 13, 13] ──────────────────────
    #
    # 这是检测头最后一个 1×1 卷积的直接输出。
    # 255 = 3 × 85，但此时 3 个 anchor 的信息是交织在一起的。

    raw_output = torch.randn(BATCH, CHANNELS, GRID_H, GRID_W)

    print(f"\n  [Step 1] 网络原始输出（检测头 1×1 卷积后）")
    print(f"           shape: {list(raw_output.shape)}")
    print(f"           [batch={BATCH}, channels={CHANNELS}, "
          f"grid_h={GRID_H}, grid_w={GRID_W}]")

    # ── Step 2: 维度变换 [B, 255, H, W] → [B, 3, 85, H, W] ────────────
    #
    # 第一步: 把 255 通道拆成 3 组，每组 85 个值
    # view() 是零拷贝操作，只改变张量的"视角"，不移动内存数据

    reshaped = raw_output.view(BATCH, ANCHORS, VALUES, GRID_H, GRID_W)

    print(f"\n  [Step 2] 通道拆分: 255 → 3 × 85")
    print(f"           shape: {list(reshaped.shape)}")
    print(f"           [batch={BATCH}, anchors={ANCHORS}, "
          f"values={VALUES}, grid_h={GRID_H}, grid_w={GRID_W}]")

    # ── Step 3: 置换维度 [B, 3, 85, H, W] → [B, H, W, 3, 85] ──────────
    #
    # 把空间维度 (H, W) 移到前面，anchor 和预测值维度移到最后。
    # 这样 [b, h, w, a, :] 就能直接索引到 "第 b 张图、(h,w) 位置、
    # 第 a 个 anchor 的 85 个预测值"，后续解码非常方便。
    #
    # permute() 也是零拷贝，只调整轴的顺序。

    final = reshaped.permute(0, 3, 4, 1, 2).contiguous()
    #                       B  H  W  A  V
    #   原始轴顺序:         0  1  2  3  4
    #   permute 后:         0  3  4  1  2

    print(f"\n  [Step 3] 维度置换: anchor/值 移到末尾")
    print(f"           permute(0, 3, 4, 1, 2)")
    print(f"           shape: {list(final.shape)}")
    print(f"           [batch={BATCH}, grid_h={GRID_H}, grid_w={GRID_W}, "
          f"anchors={ANCHORS}, values={VALUES}]")

    # ── Step 4: 验证最终形状，展示索引方式 ──────────────────────────────
    print(f"\n  [Step 4] 索引示例")
    print(f"           final[0, 5, 3, 1, :] → 第 1 张图、(5,3) 位置、"
          f"第 2 个 anchor 的 85 个值")
    print(f"           shape: {list(final[0, 5, 3, 1, :].shape)}")

    # ── Step 5: 最后一维 85 的含义 ──────────────────────────────────────
    print(f"\n  [Step 5] 最后一维 85 的含义")
    print()
    print("  索引       值          含义")
    print("  ─────────────────────────────────────────────────────")
    print("  [0]        tx         边界框中心 x 偏移量 (相对网格左上角)")
    print("  [1]        ty         边界框中心 y 偏移量")
    print("  [2]        tw         边界框宽度缩放因子 (相对 anchor 宽)")
    print("  [3]        th         边界框高度缩放因子 (相对 anchor 高)")
    print("  [4]        conf       目标置信度 (这个位置有没有物体)")
    print("  [5]~[84]   p0~p79     80 个类别的概率值 (COCO)")
    print("  ─────────────────────────────────────────────────────")
    print(f"  合计: 4 (bbox) + 1 (conf) + 80 (class) = {VALUES}")

    # ── Step 6: 一步到位的写法 ──────────────────────────────────────────
    # 实际代码中通常把 Step 2 + Step 3 合并成一行
    one_liner = raw_output.view(BATCH, ANCHORS, VALUES, GRID_H, GRID_W) \
                         .permute(0, 3, 4, 1, 2) \
                         .contiguous()

    assert one_liner.shape == final.shape, "形状应该一致"
    assert torch.equal(one_liner, final), "数值应该一致"

    print(f"\n  [Step 6] 一步到位写法（实际代码常用）")
    print(f"           output.view(B, 3, 85, H, W).permute(0,3,4,1,2).contiguous()")
    print(f"           验证: 结果与分步操作一致 ✓")

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  为什么推理解码前必须做这个维度变换？")
    print("=" * 70)
    print("""
  原始输出 [B, 255, 13, 13] 中，255 通道是连续排列的:
    ch[0]~ch[84]:   anchor 0 的 85 个值
    ch[85]~ch[169]: anchor 1 的 85 个值
    ch[170]~ch[254]:anchor 2 的 85 个值

  问题: 如果想取"位置 (5,3) 处 anchor 1 的中心 x 坐标"，需要写:
    raw[batch, 85 + 0, 5, 3]   ← 通道维度混着 anchor 索引，不直观

  reshape 后 [B, 13, 13, 3, 85]:
    final[batch, 5, 3, 1, 0]   ← 语义清晰: (y, x, anchor, tx)

  好处:
    · 解码时直接切片 final[..., 0:4] 就能拿到所有 anchor 的坐标
    · final[..., 4] 就是所有 anchor 的置信度
    · final[..., 5:] 就是所有 anchor 的类别概率
    · 后续 NMS、过滤、画框等操作都在这个维度上工作

  类比: 255 通道就像 3 个学生的成绩单被首尾相接打印在一张纸上，
  reshape 就是把它们裁开成 3 份独立的成绩单。
  """)


if __name__ == "__main__":
    main()
