"""
残差块（Residual Block）最小实验脚本
=====================================

功能：用最少的代码演示残差块的核心机制 —— shortcut（跳连）。
      输入 [1, 64, 128, 128] → 残差块 → 输出 [1, 64, 128, 128]，尺寸不变。
      不训练、不加载数据，纯结构验证。

为什么需要残差连接？（一句话版）
    让网络学习"差值" F(x) = H(x) - x，而非直接学习 H(x)。
    如果当前层不需要改动，权重可以趋向 0，输出 ≈ x，信息无损传递。

参考：He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
"""

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# 残差块：最简实现
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Darknet-53 / ResNet 风格的残差块（bottleneck 版本）。

    结构:
        输入 x ──→ 1×1 Conv (通道减半) ──→ 3×3 Conv (通道恢复) ──→ (+) ──→ 输出
          │                                                            ↑
          └──────────── shortcut（恒等映射，直通） ────────────────────┘

    参数:
        channels: 通道数。输入 = 输出 = channels，shortcut 要求三者一致。
    """

    def __init__(self, channels: int):
        super().__init__()
        half = channels // 2  # bottleneck：先压缩到一半

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, half, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(half),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(half, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x               # 保存输入，shortcut 分支
        out = self.conv1(x)        # 主分支: 1×1 压缩
        out = self.conv2(out)      # 主分支: 3×3 恢复
        return out + identity      # 残差相加: F(x) + x


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)

    # 输入张量: [batch=1, channels=64, height=128, width=128]
    x = torch.randn(1, 64, 128, 128)

    block = ResidualBlock(channels=64)
    block.eval()

    print("=" * 65)
    print("  残差块（Residual Block）结构验证")
    print("=" * 65)

    # ── 1. 打印输入尺寸 ──────────────────────────────────────────────────
    print(f"\n  [1] 输入 x        : {list(x.shape)}")

    # ── 2. 观察主分支（两条卷积路径）的输出 ─────────────────────────────
    with torch.no_grad():
        identity = x
        mid = block.conv1(x)       # 1×1 压缩后的中间结果
        main = block.conv2(mid)    # 3×3 恢复后的主分支输出

    print(f"  [2] 1×1 Conv 输出 : {list(mid.shape)}  (通道 64→32 压缩)")
    print(f"      3×3 Conv 输出 : {list(main.shape)}  (通道 32→64 恢复)")

    # ── 3. 残差相加后的最终输出 ──────────────────────────────────────────
    with torch.no_grad():
        output = block(x)

    print(f"  [3] 残差相加 output: {list(output.shape)}  (main + identity)")

    # ── 4. 尺寸一致性验证 ───────────────────────────────────────────────
    assert output.shape == x.shape, "输出尺寸必须和输入一致！"
    print(f"\n  [验证通过] output.shape == x.shape ✓")

    # ── 5. 数值验证：输出 ≠ 输入（主分支确实学到了非零变换）──────────────
    diff = (output - x).abs().mean().item()
    print(f"  output 与 x 的平均差值: {diff:.6f}")
    print(f"  (差值 ≠ 0 说明主分支产生了非零的残差变换 F(x))")

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  为什么残差连接要求输入和输出尺寸一致？")
    print("=" * 65)
    print("""
  残差连接的核心操作是逐元素相加:  output = F(x) + x

  逐元素相加要求两个张量的 shape 完全相同（batch, channel, height, width
  四个维度一一对应），否则无法相加。

  这就约束了残差块的设计:
    · 通道数不能变（输入 64 通道 → 输出必须也是 64 通道）
    · 空间尺寸不能变（输入 128×128 → 输出必须也是 128×128）
    · 所以内部用 stride=1 + padding 保持尺寸，用 bottleneck 再恢复通道数

  如果网络需要在某处改变尺寸（如下采样），怎么办？
    · 方案 A: 在残差块外部做下采样（Darknet-53 的做法）
              → 先用 stride=2 卷积改变尺寸，再进入后续残差块
    · 方案 B: 在 shortcut 路径上加一个 1×1 Conv 对齐尺寸（ResNet 原论文）
              → identity = Conv1×1(x)，然后再 output = F(x) + identity
  """)


if __name__ == "__main__":
    main()
