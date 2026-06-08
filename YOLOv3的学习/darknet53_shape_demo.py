"""
Darknet-53 Backbone 结构分析脚本（教学版）
===========================================

功能：逐层打印特征图尺寸变化，观察 416×416 如何被 Darknet-53 骨干网络
      下采样为三个关键尺度：52×52、26×26、13×13。
      不包含检测头、不加载数据集、不写训练循环。

参考：YOLOv3 论文 —— "An Incremental Improvement" (2018)

Darknet-53 结构概览（用于 416×416 输入的 YOLOv3 检测任务）：

  阶段      操作                     输出尺寸         下采样倍率
  ─────────────────────────────────────────────────────────────────
  输入      ─                        416×416×3        1×
  Stage 0   Conv 3×3(32), s=1        416×416×32       1×
  Stage 1   Conv 3×3(64), s=2        208×208×64       2×
            ResBlock ×1              208×208×64       2×
  Stage 2   Conv 3×3(128), s=2       104×104×128      4×
            ResBlock ×2              104×104×128      4×
  Stage 3   Conv 3×3(256), s=2       52×52×256        8×    ← 尺度 1: 小目标
            ResBlock ×8              52×52×256        8×
  Stage 4   Conv 3×3(512), s=2       26×26×512        16×   ← 尺度 2: 中目标
            ResBlock ×8              26×26×512        16×
  Stage 5   Conv 3×3(1024), s=2      13×13×1024       32×   ← 尺度 3: 大目标
            ResBlock ×4              13×13×1024       32×

  ※ 共 52 个卷积层构成骨干特征提取器。Darknet-53 的 "53" 来源于
     ImageNet 分类版本在 52 层卷积之后额外加了 1 层全连接层。

  ※ 与 YOLOv2 的 Darknet-19 的关键区别：
     · 用残差块（Residual Block）替代了纯堆叠卷积，网络可以更深
     · 用步长为 2 的卷积替代了最大池化层进行下采样
     · 输出三个尺度的特征图（52×52 / 26×26 / 13×13），而非仅 13×13
"""

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# 基础模块：Conv2d + BatchNorm + LeakyReLU 三件套
# Darknet-53 中每个卷积层后都紧跟 BN 和 Leaky ReLU (negative_slope=0.1)
# ═══════════════════════════════════════════════════════════════════════════════

def conv_bn_leaky(in_ch: int, out_ch: int, kernel_size: int = 3,
                  stride: int = 1) -> nn.Sequential:
    """
    Darknet-53 基础卷积单元（CBM: Conv + BN + Mish... 不对，YOLOv3 用 LeakyReLU）。

    参数:
        in_ch:       输入通道数
        out_ch:      输出通道数
        kernel_size: 卷积核大小（3 或 1）
        stride:      步长。stride=1 保持空间尺寸，stride=2 下采样 2 倍

    为什么 Darknet-53 用 stride=2 卷积替代 MaxPool 做下采样？
        · MaxPool 只取最大值，丢弃了其余信息；卷积可以学到"怎么"下采样
        · 带有可学习参数，梯度可以流过下采样层（对残差连接更友好）
        · ResNet 也是这么做的 —— Darknet-53 借鉴了 ResNet 的设计哲学
    """
    padding = 1 if kernel_size == 3 else 0
    return nn.Sequential(
        # bias=False: 因为后面紧跟 BatchNorm，BN 有自己的偏移参数 β，
        # 卷积的 bias 会被 BN 吸收，省掉无意义的冗余参数
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        # negative_slope=0.1 是 Darknet 系列的传统参数
        # 当 x < 0 时输出 0.1x 而非 0，保持微小梯度避免"神经元死亡"
        nn.LeakyReLU(0.1, inplace=True),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 残差块（Residual Block）
# 这是 Darknet-53 相比 Darknet-19 最大的结构改进
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Darknet-53 残差块（借鉴 ResNet 思想）。

    结构:
        输入 x ──→ Conv 1×1 (通道减半) ──→ Conv 3×3 (通道恢复) ──→ + ──→ 输出
          │                                                            ↑
          └──────────── shortcut（恒等映射，直接跳连）──────────────────┘

    为什么需要残差连接？
        · 深层网络面临"退化问题": 层越多训练误差反而越大（不是过拟合！）
        · 残差连接让网络只需学习"残差" F(x) = H(x) - x，即"需要改什么"
        · 如果某一层什么都不需要改，权重可以趋向 0，输出就等于输入 x
        · 这使得网络可以安全地堆叠到 50+ 层而不会退化

    为什么先 1×1 再 3×3（bottleneck 设计）？
        · 1×1 卷积先把通道数压缩一半（如 256→128），减少计算量
        · 3×3 卷积在低维空间做特征提取，再恢复原始通道数（128→256）
        · 计算量约为直接两个 3×3 卷积的 1/3，但表达能力几乎不损失

    参数:
        channels: 残差块的通道数（输入=输出，因为 shortcut 要求尺寸一致）
    """

    def __init__(self, channels: int):
        super().__init__()
        # 内部通道数减半 —— bottleneck 压缩
        half = channels // 2
        self.conv1 = conv_bn_leaky(channels, half, kernel_size=1, stride=1)
        self.conv2 = conv_bn_leaky(half, channels, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: output = F(x) + x
        F(x) 是两个卷积学到的残差映射，x 通过 shortcut 直接传递。
        """
        residual = x                              # 保存输入，作为 shortcut
        out = self.conv1(x)                       # 1×1 压缩
        out = self.conv2(out)                     # 3×3 扩张
        return out + residual                     # 残差相加 —— 核心操作！


# ═══════════════════════════════════════════════════════════════════════════════
# Darknet-53 骨干网络（Backbone）
# 输出三个尺度的特征图，供 YOLOv3 检测头使用
# ═══════════════════════════════════════════════════════════════════════════════

class Darknet53Backbone(nn.Module):
    """
    YOLOv3 使用的 Darknet-53 骨干网络。

    与 Darknet-19（YOLOv2）的对比：
    ┌──────────────┬───────────────────┬───────────────────┐
    │              │  Darknet-19       │  Darknet-53       │
    ├──────────────┼───────────────────┼───────────────────┤
    │ 核心结构     │  纯卷积堆叠       │  残差块 (ResBlock) │
    │ 下采样方式   │  MaxPool 2×2      │  Conv stride=2    │
    │ 卷积层数     │  18 层            │  52 层            │
    │ 检测特征图   │  13×13 (单尺度)   │  3 尺度 (见下方)  │
    │ 参数量       │  ~8M              │  ~23M             │
    └──────────────┴───────────────────┴───────────────────┘

    Darknet-53 输出三个尺度的特征图（这是 YOLOv3 多尺度检测的基础）：

    ┌──────────┬───────────┬────────────┬──────────────────────────────────┐
    │ 尺度名称 │ 特征图大小 │ stride倍率 │ 负责检测什么？                   │
    ├──────────┼───────────┼────────────┼──────────────────────────────────┤
    │ 尺度 1   │ 52×52     │ 8×         │ 小目标（远处的人、小物件等）      │
    │ 尺度 2   │ 26×26     │ 16×        │ 中等目标（行人、车辆等）          │
    │ 尺度 3   │ 13×13     │ 32×        │ 大目标（占据画面大部分的物体）    │
    └──────────┴───────────┴────────────┴──────────────────────────────────┘

    为什么 52×52 负责小目标、13×13 负责大目标？
        · 52×52 分辨率高 → 每个网格对应原图仅 8×8 像素 → 网格更密集
          → 能捕捉到细粒度的位置信息，适合检测小物体
        · 13×13 分辨率低 → 每个网格对应原图 32×32 像素 → 感受野更大
          → 包含更丰富的全局语义信息，适合检测大物体

    这与 FPN（Feature Pyramid Network）的思想一致：
        浅层特征图（分辨率高）→ 检测小目标
        深层特征图（分辨率低）→ 检测大目标

    YOLOv3 的完整检测流程会在此基础上做特征融合（FPN 式的自顶向下路径），
    但那属于检测头的范畴，本脚本只关注骨干网络的结构。
    """

    def __init__(self):
        super().__init__()

        # ── Stage 0: 初始卷积，416×416 保持不变 ───────────────────────────
        # 3 通道 RGB → 32 个基础特征图，网络的第一层"眼睛"
        self.stage0 = conv_bn_leaky(3, 32, kernel_size=3, stride=1)
        # 输出: 416×416×32

        # ── Stage 1: 第一次下采样 + 1 个残差块 ───────────────────────────
        # stride=2 卷积: 416→208，替代 MaxPool
        self.stage1_down = conv_bn_leaky(32, 64, kernel_size=3, stride=2)
        self.stage1_res = nn.Sequential(
            ResidualBlock(64),  # 1 个残差块
        )
        # 输出: 208×208×64

        # ── Stage 2: 第二次下采样 + 2 个残差块 ───────────────────────────
        self.stage2_down = conv_bn_leaky(64, 128, kernel_size=3, stride=2)
        self.stage2_res = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )
        # 输出: 104×104×128

        # ── Stage 3: 第三次下采样 + 8 个残差块 ───────────────────────────
        # ★★★ 关键：52×52 特征图 —— YOLOv3 尺度 1（小目标检测）★★★
        # stride=2: 104→52，下采样倍率 = 8
        # 52×52 是三个检测尺度中分辨率最高的
        self.stage3_down = conv_bn_leaky(128, 256, kernel_size=3, stride=2)
        self.stage3_res = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        # 输出: 52×52×256  ← 尺度 1

        # ── Stage 4: 第四次下采样 + 8 个残差块 ───────────────────────────
        # ★★★ 关键：26×26 特征图 —— YOLOv3 尺度 2（中目标检测）★★★
        # stride=2: 52→26，下采样倍率 = 16
        self.stage4_down = conv_bn_leaky(256, 512, kernel_size=3, stride=2)
        self.stage4_res = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        # 输出: 26×26×512  ← 尺度 2

        # ── Stage 5: 第五次下采样 + 4 个残差块 ───────────────────────────
        # ★★★ 关键：13×13 特征图 —— YOLOv3 尺度 3（大目标检测）★★★
        # stride=2: 26→13，下采样倍率 = 32
        # 最深层特征图，语义信息最丰富但空间分辨率最低
        self.stage5_down = conv_bn_leaky(512, 1024, kernel_size=3, stride=2)
        self.stage5_res = nn.Sequential(
            ResidualBlock(1024),
            ResidualBlock(1024),
            ResidualBlock(1024),
            ResidualBlock(1024),
        )
        # 输出: 13×13×1024  ← 尺度 3

    def forward(self, x: torch.Tensor, verbose: bool = True):
        """
        前向传播 + 逐层 shape 打印。

        参数:
            x:       输入张量，shape [B, 3, 416, 416]
            verbose: 是否打印每层 shape（默认 True）
        返回:
            (feat_52, feat_26, feat_13) —— 三个尺度的特征图
                feat_52: [B, 256,  52, 52]  用于小目标检测
                feat_26: [B, 512,  26, 26]  用于中目标检测
                feat_13: [B, 1024, 13, 13]  用于大目标检测
        """
        def log(name: str, tensor: torch.Tensor, note: str = ""):
            """格式化打印当前层输出的形状"""
            if verbose:
                B, C, H, W = tensor.shape
                downsample = 416 / H
                print(f"  {name:<30s} → [{B}, {C:>4d}, {H:>3d}, {W:>3d}]  "
                      f"{downsample:>4.0f}×  {note}")

        if verbose:
            print("\n" + "=" * 90)
            print("  Darknet-53 Backbone 逐层 Shape 分析")
            print("  输入尺寸: [1, 3, 416, 416]  (batch=1, RGB 三通道)")
            print("=" * 90)
            print(f"  {'层名称':<30s} → {'输出 Shape':<24s} {'下采样':>5s}  说明")
            print("-" * 90)

        # ── Stage 0: 初始特征提取 ──────────────────────────────────────────
        x = self.stage0(x)
        log("Stage0 Conv3×3(32, s=1)", x, "初始特征提取，通道 3→32")

        # ── Stage 1: 416→208 ───────────────────────────────────────────────
        x = self.stage1_down(x)
        log("Stage1 Conv3×3(64, s=2)", x, "■ 第 1 次下采样: 416→208 ■")
        x = self.stage1_res(x)
        log("Stage1 ResBlock ×1", x, "1 个残差块 (64→32→64)")

        # ── Stage 2: 208→104 ──────────────────────────────────────────────
        x = self.stage2_down(x)
        log("Stage2 Conv3×3(128, s=2)", x, "■ 第 2 次下采样: 208→104 ■")
        for i in range(2):
            x = self.stage2_res[i](x)
            log(f"Stage2 ResBlock #{i+1}", x, f"残差块 (128→64→128)")
        # 合并显示
        # log("Stage2 ResBlock ×2", x, "2 个残差块 (128→64→128)")

        # ── Stage 3: 104→52  ★★★ 尺度 1 ★★★ ─────────────────────────────
        x = self.stage3_down(x)
        log("Stage3 Conv3×3(256, s=2)", x, "■ 第 3 次下采样: 104→52 ■")
        for i in range(8):
            x = self.stage3_res[i](x)
        log("Stage3 ResBlock ×8", x, "8 个残差块 (256→128→256)")

        # ★ 保存 52×52 特征图 —— 供 YOLOv3 尺度 1 使用
        feat_52 = x
        if verbose:
            print("  " + "★" * 35)
            print(f"  ★ 尺度 1 特征图保存: {list(feat_52.shape)}"
                  f"  → 负责小目标检测 (stride=8)")
            print("  " + "★" * 35)

        # ── Stage 4: 52→26  ★★★ 尺度 2 ★★★ ─────────────────────────────
        x = self.stage4_down(x)
        log("Stage4 Conv3×3(512, s=2)", x, "■ 第 4 次下采样: 52→26 ■")
        for i in range(8):
            x = self.stage4_res[i](x)
        log("Stage4 ResBlock ×8", x, "8 个残差块 (512→256→512)")

        # ★ 保存 26×26 特征图 —— 供 YOLOv3 尺度 2 使用
        feat_26 = x
        if verbose:
            print("  " + "★" * 35)
            print(f"  ★ 尺度 2 特征图保存: {list(feat_26.shape)}"
                  f"  → 负责中目标检测 (stride=16)")
            print("  " + "★" * 35)

        # ── Stage 5: 26→13  ★★★ 尺度 3 ★★★ ─────────────────────────────
        x = self.stage5_down(x)
        log("Stage5 Conv3×3(1024, s=2)", x, "■ 第 5 次下采样: 26→13 ■")
        for i in range(4):
            x = self.stage5_res[i](x)
        log("Stage5 ResBlock ×4", x, "4 个残差块 (1024→512→1024)")

        # ★ 保存 13×13 特征图 —— 供 YOLOv3 尺度 3 使用
        feat_13 = x
        if verbose:
            print("  " + "★" * 35)
            print(f"  ★ 尺度 3 特征图保存: {list(feat_13.shape)}"
                  f"  → 负责大目标检测 (stride=32)")
            print("  " + "★" * 35)

        if verbose:
            print("-" * 90)
            print("  [OK] Darknet-53 Backbone 输出三个尺度的特征图:")
            print(f"    feat_52: {list(feat_52.shape)}  → stride=8,  小目标")
            print(f"    feat_26: {list(feat_26.shape)}  → stride=16, 中目标")
            print(f"    feat_13: {list(feat_13.shape)}  → stride=32, 大目标")
            print("=" * 90 + "\n")

        return feat_52, feat_26, feat_13


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序：创建 dummy 输入，跑一遍前向，观察所有 shape 变化
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # 设置随机种子，保证每次运行输出一致
    torch.manual_seed(42)

    # 1. 创建模型（eval 模式，BN 用运行均值/方差，不计算梯度）
    model = Darknet53Backbone()
    model.eval()

    # 2. 构造一张假输入图: batch=1, RGB=3, 416×416
    dummy_input = torch.randn(1, 3, 416, 416)

    # 3. 统计总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Darknet-53 Backbone 总参数量: {total_params:,}  "
          f"(约 {total_params/1e6:.1f} M)")

    # 对比 Darknet-19
    print(f"  (对比: Darknet-19 Backbone 约 8.0 M，Darknet-53 是其 "
          f"{total_params/8e6:.1f} 倍)")

    # 4. 前向传播（逐层打印 shape）—— 不需要梯度
    with torch.no_grad():
        feat_52, feat_26, feat_13 = model(dummy_input, verbose=True)

    # 5. 验证输出尺寸
    assert feat_52.shape[2:] == (52, 52), f"尺度 1 应为 52×52，实际 {feat_52.shape}"
    assert feat_26.shape[2:] == (26, 26), f"尺度 2 应为 26×26，实际 {feat_26.shape}"
    assert feat_13.shape[2:] == (13, 13), f"尺度 3 应为 13×13，实际 {feat_13.shape}"
    print(f"  [验证通过] 三个尺度特征图尺寸正确: 52×52, 26×26, 13×13\n")

    # ═══════════════════════════════════════════════════════════════════════
    # 补充说明：为什么 YOLOv3 需要三个尺度？
    # ═══════════════════════════════════════════════════════════════════════
    print("-" * 90)
    print("  [*] 为什么 YOLOv3 需要三个尺度的特征图？")
    print()
    print("  1. YOLOv2 的痛点:")
    print("     Darknet-19 只输出 13×13 特征图（stride=32），每个网格覆盖原图 32×32 像素。")
    print("     对于小目标（如远处的行人），32×32 的分辨率太粗，容易漏检。")
    print()
    print("  2. YOLOv3 的解决方案 —— 多尺度检测（借鉴 FPN 思想）:")
    print()
    print("     ┌─────────────────────────────────────────────────────────┐")
    print("     │  尺度 1:  52×52×256   stride=8    →  检测小目标        │")
    print("     │  每个网格覆盖 8×8 像素，分辨率高，能看到细小物体      │")
    print("     │  例: 远处的行人、手中的手机、桌上的杯子                 │")
    print("     ├─────────────────────────────────────────────────────────┤")
    print("     │  尺度 2:  26×26×512   stride=16   →  检测中目标        │")
    print("     │  每个网格覆盖 16×16 像素，平衡了分辨率和语义信息       │")
    print("     │  例: 正常距离的行人、停放的汽车、椅子                   │")
    print("     ├─────────────────────────────────────────────────────────┤")
    print("     │  尺度 3:  13×13×1024  stride=32   →  检测大目标        │")
    print("     │  每个网格覆盖 32×32 像素，感受野大，语义理解深         │")
    print("     │  例: 占满画面的人、大型车辆、整栋建筑                   │")
    print("     └─────────────────────────────────────────────────────────┘")
    print()
    print("  3. 每个尺度分配 3 个 anchor（通过 K-means 聚类得到），共 9 个 anchor：")
    print("     · 尺度 1 (52×52): 小 anchor (10×13, 16×30, 33×23)")
    print("     · 尺度 2 (26×26): 中 anchor (30×61, 62×45, 59×119)")
    print("     · 尺度 3 (13×13): 大 anchor (116×90, 156×198, 373×326)")
    print()
    print("  4. 检测头数量计算（以 COCO 80 类为例）:")
    print("     每个网格 × 3 anchor × (5 + 80) = 255 通道")
    print("     · 52×52: 52×52×3 = 8,112 个预测框")
    print("     · 26×26: 26×26×3 = 2,028 个预测框")
    print("     · 13×13: 13×13×3 =   507 个预测框")
    print("     · 总计: 10,647 个预测框 (vs YOLOv2 的 845 个)")
    print()
    print("  ─────────────────────────────────────────────────────────")
    print("  ★ 总结:")
    print("    Darknet-53 相比 Darknet-19 的核心改进:")
    print("    · 残差连接 → 网络可以更深（52 层 vs 18 层），特征提取能力更强")
    print("    · 三尺度输出 → 小/中/大目标都有专门的特征图负责检测")
    print("    · stride 卷积下采样 → 比 MaxPool 更可学习，对梯度流更友好")
    print()
    print("    三个尺度的特征图之所以能分别负责不同大小的目标，根本原因是")
    print("    它们在「空间分辨率」和「感受野」之间做了不同的权衡:")
    print("    高分辨率 (52×52) → 定位精准 → 小目标")
    print("    低分辨率  (13×13) → 语义丰富 → 大目标")
    print("  ─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
