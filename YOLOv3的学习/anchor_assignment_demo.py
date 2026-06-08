"""
YOLOv3 Anchor Boxes 分配演示脚本
=================================

功能：计算 9 个 anchor 的面积，按大小排序后分配到三个检测尺度。
      不依赖 PyTorch，纯 Python + NumPy 演示。

YOLOv3 的 9 个 anchor（来自 COCO 数据集 K-means 聚类）:
    (10,13), (16,30), (33,23), (30,61), (62,45),
    (59,119), (116,90), (156,198), (373,326)

    注意: 这些宽高是相对于 416×416 输入图片的像素值，不是 grid cell 单位。

分配规则（面积从小到大，每 3 个一组）:
    · 最小 3 个 → 52×52 尺度 (stride=8)  → 小目标
    · 中间 3 个 → 26×26 尺度 (stride=16) → 中目标
    · 最大 3 个 → 13×13 尺度 (stride=32) → 大目标

参考：YOLOv3 论文 —— "An Incremental Improvement" (2018)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 9 个 Anchor（宽, 高），单位: 像素（相对于 416×416 输入）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 这些值是通过在 COCO 训练集上做 K-means 聚类得到的（k=9），
# 聚类距离使用 IoU 距离 d(box, centroid) = 1 - IoU(box, centroid)，
# 而非欧氏距离，这样对框的绝对大小不敏感。

ANCHORS = [
    (10,  13),
    (16,  30),
    (33,  23),
    (30,  61),
    (62,  45),
    (59, 119),
    (116, 90),
    (156, 198),
    (373, 326),
]

# 三个检测尺度
SCALES = [
    {"name": "尺度 1（小目标）", "grid": 52, "stride":  8},
    {"name": "尺度 2（中目标）", "grid": 26, "stride": 16},
    {"name": "尺度 3（大目标）", "grid": 13, "stride": 32},
]


def main():
    print("=" * 70)
    print("  YOLOv3 Anchor Boxes 分配演示")
    print("=" * 70)

    # ── Step 1: 计算每个 anchor 的面积 ──────────────────────────────────
    print(f"\n  [Step 1] 9 个 Anchor 的面积")

    anchors_with_area = []
    for i, (w, h) in enumerate(ANCHORS):
        area = w * h
        anchors_with_area.append({"id": i + 1, "w": w, "h": h, "area": area})

    print(f"  {'序号':>4s}  {'宽(w)':>6s}  {'高(h)':>6s}  {'面积':>8s}")
    print("  " + "-" * 32)
    for a in anchors_with_area:
        print(f"  {a['id']:>4d}  {a['w']:>6d}  {a['h']:>6d}  {a['area']:>8d}")

    # ── Step 2: 按面积从小到大排序 ──────────────────────────────────────
    sorted_anchors = sorted(anchors_with_area, key=lambda a: a["area"])

    print(f"\n  [Step 2] 按面积排序（小 → 大）")
    print(f"  {'排名':>4s}  {'原序号':>6s}  {'宽':>6s}  {'高':>6s}  {'面积':>8s}")
    print("  " + "-" * 38)
    for rank, a in enumerate(sorted_anchors, 1):
        print(f"  {rank:>4d}  #{a['id']:<5d} {a['w']:>5d}  {a['h']:>5d}  "
              f"{a['area']:>8d}")

    # ── Step 3: 分配到三个尺度（每 3 个一组）───────────────────────────
    #
    # YOLOv3 原始配置文件中 anchor 的排列顺序:
    #   前 3 个给大尺度（13×13），中间 3 个给中尺度（26×26），后 3 个给小尺度（52×52）
    # 但按面积理解更直观: 小 anchor → 小目标 → 高分辨率特征图

    groups = [
        sorted_anchors[0:3],   # 最小 3 个 → 52×52
        sorted_anchors[3:6],   # 中间 3 个 → 26×26
        sorted_anchors[6:9],   # 最大 3 个 → 13×13
    ]

    print(f"\n  [Step 3] 分配到三个检测尺度")
    print("  " + "=" * 62)

    for scale, group in zip(SCALES, groups):
        print(f"\n  {scale['name']}  "
              f"grid={scale['grid']}×{scale['grid']}  stride={scale['stride']}")
        print(f"  {'anchor':>8s}  {'宽':>6s}  {'高':>6s}  {'面积':>8s}  "
              f"{'宽/stride':>10s}")
        print("  " + "-" * 50)

        for a in group:
            # 宽高除以 stride，得到在 grid cell 单位下的尺寸
            cell_w = a["w"] / scale["stride"]
            cell_h = a["h"] / scale["stride"]
            print(f"  #{a['id']:<6d}  {a['w']:>6d}  {a['h']:>6d}  "
                  f"{a['area']:>8d}  {cell_w:>5.1f}×{cell_h:<4.1f} cells")

    # ── Step 4: 分配总览 ──────────────────────────────────────────────
    print(f"\n  [Step 4] 分配总览")
    print()
    print("  尺度              stride   anchor (像素)                        面积范围")
    print("  " + "-" * 66)

    for scale, group in zip(SCALES, groups):
        areas = [a["area"] for a in group]
        anchor_strs = [f"({a['w']},{a['h']})" for a in group]
        print(f"  {scale['name']:<16s}{scale['stride']:>4d}   "
              f"{', '.join(anchor_strs):<30s}  "
              f"{min(areas):>6d} ~ {max(areas):<6d}")

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  为什么小 Anchor 对应小目标检测层（52×52）？")
    print("=" * 70)
    print("""
  1. 匹配逻辑:
     · 小目标在图片中占的像素少（如 20×30 像素），需要小尺寸的 anchor 去匹配
     · 如果用大 anchor (373×326) 去匹配小目标，IoU 会非常低
     · 反过来，大 anchor 匹配大目标时 IoU 才高

  2. 为什么小 anchor 放在高分辨率特征图（52×52）？
     · 52×52 特征图的每个网格只覆盖原图 8×8 像素（stride=8）
     · 网格密集 → 即使目标只占很小区域，也能被至少一个网格覆盖到
     · 小 anchor + 密集网格 = 捕捉小目标的最佳组合

     · 13×13 特征图的每个网格覆盖 32×32 像素（stride=32）
     · 网格稀疏 → 小目标可能直接落在网格间隙中，根本不会被"看到"
     · 但大 anchor 放在这里没问题，因为大目标不容易被遗漏

  3. 一个直观的例子:
     假设图片中远处有一个人，只有 15×20 像素:

     · 52×52 层: stride=8, 这个小人跨越约 2×3 个网格
       用 anchor (10,13) 匹配 → IoU 较高 ✓
     · 13×13 层: stride=32, 这个小人不到 1 个网格大小
       用 anchor (373,326) 匹配 → 完全不匹配 ✗

     所以: 小 anchor 必须配高分辨率层，否则小目标无 anchor 可用。

  4. 面积跨度直观感受:
     最小 anchor: 10×13 =    130 像素²  (小如拳头)
     最大 anchor: 373×326 = 121,598 像素²  (占满大半个画面)
     相差约 935 倍 → 覆盖了从极小到极大的目标尺寸范围
    """)


if __name__ == "__main__":
    main()
