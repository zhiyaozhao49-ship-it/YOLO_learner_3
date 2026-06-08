"""
YOLOv3 Target Assignment（目标分配）教学脚本
==============================================

功能：演示 YOLOv3 训练时如何将每个 Ground Truth 分配到
      具体的尺度 + anchor + 网格位置。

      不训练、不推理，纯算法演示。

YOLOv3 Target Assignment 流程:
    1. 将 GT 框的宽高（像素单位）与 9 个 anchor 计算宽高 IoU（不考虑位置）
    2. 给每个 GT 分配 IoU 最高的 anchor
    3. 根据 anchor 所属组确定负责检测该目标的尺度（13×13 / 26×26 / 52×52）
    4. 根据 GT 中心坐标确定具体落在哪个网格 (grid_y, grid_x)

参考：YOLOv3 论文 —— "An Incremental Improvement" (2018)
"""

import math


# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_SIZE = 416  # 输入图片边长

# 9 个 anchor（宽, 高），单位: 像素，相对 416×416
# 通过 COCO 数据集 K-means 聚类得到
ANCHORS = [
    (10,  13),   # 小 #0
    (16,  30),   # 小 #1
    (33,  23),   # 小 #2
    (30,  61),   # 中 #3
    (62,  45),   # 中 #4
    (59, 119),   # 中 #5
    (116, 90),   # 大 #6
    (156,198),   # 大 #7
    (373,326),   # 大 #8
]

# anchor 分组: 按面积从小到大，每 3 个一组
# 排序后: #0(130) < #1(480) < #2(759) < #3(1830) < #4(2790) < #5(7021)
#         < #6(10440) < #7(30888) < #8(121598)
ANCHOR_GROUPS = {
    "52×52 (stride=8, 小目标)":  [0, 1, 2],   # 最小 3 个
    "26×26 (stride=16, 中目标)": [3, 4, 5],   # 中间 3 个
    "13×13 (stride=32, 大目标)": [6, 7, 8],   # 最大 3 个
}

# 模拟 Ground Truth（真实标注框）
# 格式: (class_id, x_center, y_center, width, height)
# 坐标单位: 像素，相对 416×416
GROUND_TRUTH = [
    {"name": "远处行人", "class": 0,  "cx": 200, "cy": 150, "w":  15, "h":  40},
    {"name": "近处行人", "class": 0,  "cx": 300, "cy": 300, "w":  60, "h": 180},
    {"name": "汽车",    "class": 2,  "cx": 100, "cy": 250, "w": 150, "h": 100},
    {"name": "小猫",    "class": 16, "cx": 350, "cy":  50, "w":  20, "h":  18},
    {"name": "大卡车",  "class": 7,  "cx": 208, "cy": 208, "w": 380, "h": 300},
]


# ═══════════════════════════════════════════════════════════════════════════════
# 宽高 IoU 计算（不考虑位置，只比较形状匹配度）
# ═══════════════════════════════════════════════════════════════════════════════

def wh_iou(box_w: float, box_h: float, anchor_w: float, anchor_h: float) -> float:
    """
    计算两个矩形"只看宽高"的 IoU（忽略位置差异）。

    为什么只看宽高？
        · anchor 的位置由网格决定，不是自由变量
        · Target Assignment 只关心"哪个 anchor 的形状最适合这个 GT"
        · 所以把 GT 和 anchor 都"平移"到原点对齐，只比较面积重叠

    数学:
        intersection = min(w1, w2) × min(h1, h2)
        union        = w1×h1 + w2×h2 - intersection
        IoU          = intersection / union

    例: GT=(30,40), anchor=(32,45)
        intersection = 30×40 = 1200
        union        = 30×40 + 32×45 - 1200 = 1200 + 1440 - 1200 = 1440
        IoU          = 1200/1440 ≈ 0.833
    """
    inter_w = min(box_w, anchor_w)
    inter_h = min(box_h, anchor_h)
    intersection = inter_w * inter_h
    union = box_w * box_h + anchor_w * anchor_h - intersection
    if union <= 0:
        return 0.0
    return intersection / union


# ═══════════════════════════════════════════════════════════════════════════════
# 查找 anchor 所属的尺度
# ═══════════════════════════════════════════════════════════════════════════════

def find_scale(anchor_idx: int) -> tuple:
    """
    根据 anchor 索引返回所属尺度的信息。
    返回: (尺度名称, stride, grid_size)
    """
    scale_info = [
        ("52×52", 8,  52),
        ("26×26", 16, 26),
        ("13×13", 32, 13),
    ]
    for (name, stride, grid), indices in zip(scale_info, ANCHOR_GROUPS.values()):
        if anchor_idx in indices:
            return name, stride, grid
    return "未知", 0, 0


def main():
    print("=" * 75)
    print("  YOLOv3 Target Assignment（目标分配）演示")
    print("=" * 75)

    # ── Step 1: 展示 Ground Truth ──────────────────────────────────────
    print(f"\n  [Step 1] Ground Truth 真实标注框 (图片 {INPUT_SIZE}×{INPUT_SIZE})")
    print(f"  {'目标':>10s}  {'class':>5s}  {'cx':>5s}  {'cy':>5s}  "
          f"{'w':>5s}  {'h':>5s}  {'面积':>8s}")
    print("  " + "-" * 55)
    for gt in GROUND_TRUTH:
        area = gt["w"] * gt["h"]
        print(f"  {gt['name']:>10s}  {gt['class']:>5d}  {gt['cx']:>5d}  "
              f"{gt['cy']:>5d}  {gt['w']:>5d}  {gt['h']:>5d}  {area:>8d}")

    # ── Step 2: 计算每个 GT 与 9 个 anchor 的宽高 IoU ─────────────────
    print(f"\n  [Step 2] 宽高 IoU 矩阵（GT × Anchors）")
    print()

    # 打印表头
    header = f"  {'':>10s}"
    for i, (aw, ah) in enumerate(ANCHORS):
        header += f"  #{i}({aw},{ah})"
    print(header)
    print("  " + "-" * (12 + 11 * 9))

    # 存储分配结果
    assignments = []

    for gt in GROUND_TRUTH:
        row = f"  {gt['name']:>10s}"
        ious = []
        for i, (aw, ah) in enumerate(ANCHORS):
            iou = wh_iou(gt["w"], gt["h"], aw, ah)
            ious.append(iou)
            row += f"  {iou:>8.3f}"
        print(row)

        # 找 IoU 最高的 anchor
        best_idx = ious.index(max(ious))
        best_iou = ious[best_idx]
        scale_name, stride, grid = find_scale(best_idx)

        # 计算目标落在哪个网格
        grid_x = int(gt["cx"] / stride)
        grid_y = int(gt["cy"] / stride)
        # 防止越界
        grid_x = min(grid_x, grid - 1)
        grid_y = min(grid_y, grid - 1)

        assignments.append({
            "gt": gt,
            "best_anchor": best_idx,
            "best_iou": best_iou,
            "scale": scale_name,
            "stride": stride,
            "grid": grid,
            "grid_x": grid_x,
            "grid_y": grid_y,
        })

    # ── Step 3: 打印分配结果 ───────────────────────────────────────────
    print(f"\n  [Step 3] 分配结果")
    print("  " + "=" * 70)
    print(f"  {'目标':>10s}  {'最佳anchor':>10s}  {'IoU':>6s}  "
          f"{'负责尺度':>10s}  {'网格位置':>10s}  {'anchor像素':>12s}")
    print("  " + "-" * 70)

    for a in assignments:
        gt = a["gt"]
        aw, ah = ANCHORS[a["best_anchor"]]
        print(f"  {gt['name']:>10s}  #{a['best_anchor']:>9d}  "
              f"{a['best_iou']:>6.3f}  {a['scale']:>10s}  "
              f"({a['grid_y']},{a['grid_x']:>2d}){'':>5s}  "
              f"({aw},{ah})")

    # ── Step 4: 逐个目标的详细分析 ─────────────────────────────────────
    print(f"\n  [Step 4] 逐目标详细分析")
    print("  " + "=" * 70)

    for a in assignments:
        gt = a["gt"]
        aw, ah = ANCHORS[a["best_anchor"]]
        print(f"\n  ── {gt['name']} (class={gt['class']}, "
              f"大小={gt['w']}×{gt['h']}, 面积={gt['w']*gt['h']})")
        print(f"     GT 中心: ({gt['cx']}, {gt['cy']}) 像素")
        print(f"     最佳 anchor: #{a['best_anchor']} ({aw}, {ah}), "
              f"宽高 IoU = {a['best_iou']:.3f}")
        print(f"     → 分配到尺度: {a['scale']} (stride={a['stride']})")
        print(f"     → 负责网格: ({a['grid_y']}, {a['grid_x']}) "
              f"/ {a['grid']}×{a['grid']}")

        # 计算 GT 在 cell 单位下的宽高
        cell_w = gt["w"] / a["stride"]
        cell_h = gt["h"] / a["stride"]
        print(f"     → GT 在 cell 单位: 宽={cell_w:.2f}, 高={cell_h:.2f} "
              f"(anchor cell: {aw/a['stride']:.2f}, {ah/a['stride']:.2f})")

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("  为什么 Target Assignment 会影响训练效果？")
    print("=" * 75)
    print("""
  1. 每个 GT 必须有"老师"来教:
     · 训练时，网络需要知道"哪个位置的哪个 anchor 应该预测这个目标"
     · Target Assignment 就是把"学生"（anchor 位置）和"教材"（GT 框）匹配起来
     · 如果分配错误（比如把小目标分给大 anchor），网络就学不好

  2. 宽高 IoU 分配保证了"因材施教":
     · 高 IoU → anchor 形状和目标形状匹配 → 只需要微调 → 容易学
     · 低 IoU → anchor 形状和目标形状差很远 → 需要大幅调整 → 难学
     · 例如: 一个 15×40 的行人，分配给 anchor(10,13) 时 IoU 很低
       但分配给 anchor(16,30) 时 IoU 更高，网络更容易收敛

  3. 分配决定梯度的"流向":
     · 被分配的 anchor 位置 → 正样本 → 产生坐标回归梯度 + 分类梯度
     · 未被分配的 anchor 位置 → 负样本 → 只产生置信度梯度（"这里没东西"）
     · 如果一个 GT 被错误分配到 13×13 尺度，那 52×52 尺度永远学不到
       检测这类目标 → 推理时就会漏检

  4. 本例中的直观感受:
     · "远处行人"(15×40) → 分配到 52×52 尺度 → 合理，小目标由高分辨率层负责
     · "大卡车"(380×300) → 分配到 13×13 尺度 → 合理，超大目标由低分辨率层负责
     · 如果反过来分配，大卡车在 52×52 层会覆盖几十个网格，难以精确定位；
       远处行人在 13×13 层不到半个网格，根本无法匹配

  5. YOLOv3 vs YOLOv2 的分配改进:
     · YOLOv2 只有 5 个 anchor 在 13×13 单尺度上，小目标难以被分配到
     · YOLOv3 有 9 个 anchor 分布在 3 个尺度，每种大小的目标都能找到
       形状匹配的 anchor + 分辨率合适的特征图
    """)


if __name__ == "__main__":
    main()
