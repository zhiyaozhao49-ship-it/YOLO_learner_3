"""
YOLOv3 后处理教学脚本（置信度过滤 + NMS）
============================================

功能：模拟 YOLOv3 推理后的完整后处理流程:
      1. 计算最终得分 = objectness × class_score
      2. 置信度阈值过滤（去掉低分框）
      3. NMS 非极大值抑制（去掉重复框）

      不需要真实模型，随机生成预测框演示。

YOLOv3 后处理流水线:
    网络输出 (10647 个预测框)
        ↓ 解码得到 (x1, y1, x2, y2)
        ↓ 计算 final_score = objectness × class_score
        ↓ 置信度阈值过滤 (如 < 0.5 则丢弃)
        ↓ 按类别分组
        ↓ 对每个类别独立做 NMS
        ↓ 最终检测结果

参考：YOLOv3 论文 —— "An Incremental Improvement" (2018)
"""

import random


# ═══════════════════════════════════════════════════════════════════════════════
# 配置参数
# ═══════════════════════════════════════════════════════════════════════════════

CONF_THRESHOLD = 0.5   # 置信度阈值: 低于此值的框直接丢弃
NMS_IOU_THRESHOLD = 0.45  # NMS 的 IoU 阈值: 重叠超过此值的框被抑制
NUM_BOXES = 30         # 模拟的预测框数量（代表一张图的全部预测）
IMAGE_SIZE = 416       # 图片尺寸

# 简化: 只用 3 个类别做演示
CLASSES = {0: "人", 1: "汽车", 2: "猫"}


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def compute_iou(box_a: dict, box_b: dict) -> float:
    """
    计算两个框的 IoU (Intersection over Union, 交并比)。

    IoU = 交集面积 / 并集面积

    取值范围 [0, 1]:
        0   → 完全不重叠
        1   → 完全重合
        0.5 → 有一半区域重叠

    这是目标检测中最核心的度量，用于:
        · NMS: IoU 高 → 两个框检测的是同一个物体 → 删掉分数低的
        · mAP: IoU > 阈值 → 检测正确（TP），否则是误检（FP）
    """
    # 交集的左上角和右下角
    inter_x1 = max(box_a["x1"], box_b["x1"])
    inter_y1 = max(box_a["y1"], box_b["y1"])
    inter_x2 = min(box_a["x2"], box_b["x2"])
    inter_y2 = min(box_a["y2"], box_b["y2"])

    # 交集面积（宽高不能为负）
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 各自的面积
    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])

    # 并集面积 = A + B - 交集
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def generate_boxes(num: int) -> list:
    """
    随机生成预测框，模拟 YOLOv3 解码后的输出。

    为了让 NMS 有意义，故意在 3 个"聚类中心"附近生成框，
    模拟多个 anchor 检测到同一个物体的常见场景。
    """
    random.seed(7)
    boxes = []

    # 3 个"物体"的近似位置，周围会聚拢多个预测框
    clusters = [
        {"cx": 100, "cy": 120, "w": 60,  "h": 160, "cls": 0},   # 人
        {"cx": 300, "cy": 300, "w": 140, "h": 90,  "cls": 1},   # 汽车
        {"cx": 350, "cy":  60, "w": 30,  "h": 25,  "cls": 2},   # 猫
    ]

    for i in range(num):
        # 随机选一个聚类中心
        c = random.choice(clusters)

        # 在中心附近加随机偏移，模拟不同 anchor 的预测差异
        # 偏移较小 → 多个框聚集在同一区域 → NMS 有意义
        cx = c["cx"] + random.randint(-15, 15)
        cy = c["cy"] + random.randint(-15, 15)
        w  = c["w"]  + random.randint(-10, 10)
        h  = c["h"]  + random.randint(-10, 10)

        # 限制在图片范围内
        w = max(10, min(w, IMAGE_SIZE))
        h = max(10, min(h, IMAGE_SIZE))
        x1 = max(0, cx - w // 2)
        y1 = max(0, cy - h // 2)
        x2 = min(IMAGE_SIZE, x1 + w)
        y2 = min(IMAGE_SIZE, y1 + h)

        # 随机生成 objectness 和 class_score
        objectness = random.uniform(0.1, 1.0)
        class_score = random.uniform(0.2, 1.0)

        boxes.append({
            "id": i,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "objectness": round(objectness, 3),
            "class_score": round(class_score, 3),
            "class_id": c["cls"],
        })

    return boxes


def nms(boxes: list, iou_threshold: float) -> list:
    """
    Non-Maximum Suppression（非极大值抑制）。

    算法流程（每个类别独立执行）:
        1. 按分数从高到低排序
        2. 取分数最高的框 A，加入结果列表
        3. 计算剩余所有框与 A 的 IoU
        4. IoU > 阈值的框视为"重复检测"，删除
        5. 重复步骤 2~4 直到没有框剩余

    为什么每个类别独立做？
        · 一个框检测"人"，另一个框检测"汽车"，即使位置重叠也不该被抑制
        · 只有"同类别 + 高重叠"才是重复检测
    """
    if not boxes:
        return []

    # 按 final_score 降序排列
    boxes_sorted = sorted(boxes, key=lambda b: b["final_score"], reverse=True)

    kept = []

    while boxes_sorted:
        # 取当前分数最高的框
        best = boxes_sorted.pop(0)
        kept.append(best)

        # 剩余框中，与 best 的 IoU 超过阈值的删除
        remaining = []
        for box in boxes_sorted:
            # 只对同类别做 NMS（不同类别不会被抑制）
            if box["class_id"] != best["class_id"]:
                remaining.append(box)
                continue

            iou = compute_iou(best, box)
            if iou < iou_threshold:
                remaining.append(box)
            # IoU >= 阈值 → 重复框，丢弃（不放入 remaining）

        boxes_sorted = remaining

    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 75)
    print("  YOLOv3 后处理流水线（置信度过滤 + NMS）")
    print(f"  置信度阈值: {CONF_THRESHOLD}   NMS IoU 阈值: {NMS_IOU_THRESHOLD}")
    print("=" * 75)

    # ── Step 1: 生成模拟预测框 ──────────────────────────────────────────
    boxes = generate_boxes(NUM_BOXES)

    print(f"\n  [Step 1] 模拟网络输出: {len(boxes)} 个预测框")
    print(f"  (实际 YOLOv3 416×416 输入会产生 10,647 个预测框)")
    print()
    print(f"  {'id':>4s}  {'x1':>5s}  {'y1':>5s}  {'x2':>5s}  {'y2':>5s}  "
          f"{'obj':>6s}  {'cls_s':>6s}  {'class':>6s}")
    print("  " + "-" * 55)
    for b in boxes[:10]:  # 只打印前 10 个
        print(f"  {b['id']:>4d}  {b['x1']:>5d}  {b['y1']:>5d}  "
              f"{b['x2']:>5d}  {b['y2']:>5d}  "
              f"{b['objectness']:>6.3f}  {b['class_score']:>6.3f}  "
              f"{CLASSES[b['class_id']]:>4s}")
    print(f"  ... 共 {len(boxes)} 个框（省略部分输出）")

    # ── Step 2: 计算最终得分 ────────────────────────────────────────────
    #
    # final_score = objectness × class_score
    #
    # 为什么是两个分数相乘？
    #   · objectness: "这个位置有没有物体？"（0~1）
    #   · class_score: "如果有物体，是这个类别的概率？"（0~1）
    #   · 相乘 = "这个位置有这个类别物体的综合概率"
    #
    # 两个维度都必须高，最终得分才会高:
    #   objectness=0.9, class=0.9 → final=0.81 ✓
    #   objectness=0.1, class=0.9 → final=0.09 ✗ (背景，不是猫)
    #   objectness=0.9, class=0.1 → final=0.09 ✗ (有物体但不是这个类)

    print(f"\n  [Step 2] 计算 final_score = objectness × class_score")

    for b in boxes:
        b["final_score"] = round(b["objectness"] * b["class_score"], 4)

    # 打印分数分布
    scores = [b["final_score"] for b in boxes]
    print(f"           分数范围: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"           平均分数: {sum(scores)/len(scores):.4f}")

    # ── Step 3: 置信度过滤 ──────────────────────────────────────────────
    #
    # YOLOv3 产生 10,647 个预测框，绝大多数是背景（没有物体）
    # 置信度过滤是最粗暴也最高效的第一道筛子:
    #   final_score < threshold → 直接丢弃，不做后续处理

    filtered = [b for b in boxes if b["final_score"] >= CONF_THRESHOLD]

    print(f"\n  [Step 3] 置信度过滤 (threshold={CONF_THRESHOLD})")
    print(f"           过滤前: {len(boxes)} 个框")
    print(f"           过滤后: {len(filtered)} 个框")
    print(f"           丢弃:   {len(boxes) - len(filtered)} 个框 "
          f"({(1 - len(filtered)/len(boxes))*100:.0f}%)")

    if not filtered:
        print("\n  所有框都被过滤，流程结束。")
        return

    # 打印过滤后的框
    print()
    print(f"  {'id':>4s}  {'位置(x1,y1,x2,y2)':>22s}  "
          f"{'obj':>5s}  {'cls':>5s}  {'final':>6s}  {'类别':>4s}")
    print("  " + "-" * 62)
    for b in sorted(filtered, key=lambda x: x["final_score"], reverse=True):
        print(f"  {b['id']:>4d}  ({b['x1']:>3d},{b['y1']:>3d},{b['x2']:>3d},{b['y2']:>3d})"
              f"       {b['objectness']:>5.3f}  {b['class_score']:>5.3f}  "
              f"{b['final_score']:>6.4f}  {CLASSES[b['class_id']]:>4s}")

    # ── Step 4: NMS 非极大值抑制 ────────────────────────────────────────
    #
    # 即使过了置信度过滤，同一物体仍可能有多个高分框:
    #   · 不同 anchor 匹配到同一物体 → 多个框重叠
    #   · 相邻网格都检测到同一物体 → 框几乎重合
    # NMS 的作用: 每个物体只保留分数最高的一个框

    print(f"\n  [Step 4] NMS 非极大值抑制 (IoU threshold={NMS_IOU_THRESHOLD})")
    print(f"           输入: {len(filtered)} 个框")

    # 按类别分组，每个类别独立做 NMS
    results = []
    for cls_id in CLASSES:
        cls_boxes = [b for b in filtered if b["class_id"] == cls_id]
        if not cls_boxes:
            continue
        kept = nms(cls_boxes, NMS_IOU_THRESHOLD)
        results.extend(kept)

    print(f"           NMS 后: {len(results)} 个框")
    print(f"           抑制:   {len(filtered) - len(results)} 个重复框")

    # 打印最终结果
    print()
    print("  " + "=" * 62)
    print(f"  最终检测结果（{len(results)} 个物体）:")
    print("  " + "=" * 62)
    print(f"  {'id':>4s}  {'位置(x1,y1,x2,y2)':>22s}  "
          f"{'score':>6s}  {'类别':>4s}  {'大小':>12s}")
    print("  " + "-" * 62)
    for b in sorted(results, key=lambda x: x["final_score"], reverse=True):
        w = b["x2"] - b["x1"]
        h = b["y2"] - b["y1"]
        print(f"  {b['id']:>4d}  ({b['x1']:>3d},{b['y1']:>3d},{b['x2']:>3d},{b['y2']:>3d})"
              f"       {b['final_score']:>6.4f}  {CLASSES[b['class_id']]:>4s}  "
              f"{w}×{h}")

    # ── 汇总统计 ────────────────────────────────────────────────────────
    print(f"\n  ─── 后处理流水线汇总 ───")
    print(f"  原始预测框:     {NUM_BOXES:>4d}")
    print(f"  置信度过滤后:   {len(filtered):>4d}  "
          f"(丢弃 {NUM_BOXES - len(filtered)})")
    print(f"  NMS 后:         {len(results):>4d}  "
          f"(抑制 {len(filtered) - len(results)})")

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("  为什么 YOLOv3 必须做 NMS？")
    print("=" * 75)
    print("""
  1. 同一物体会产生多个预测框:
     · YOLOv3 有 10,647 个预测框，每个框独立预测
     · 一个物体可能同时被多个 anchor / 多个网格检测到
     · 例如: 一个人可能被周围的 3~5 个网格同时预测出来
     · 没有后处理 → 画面上同一个位置叠着 5 个框 → 用户体验差

  2. 置信度过滤只能去掉"背景框"，无法去重:
     · 多个框检测同一物体 → 分数都很高 → 都通过了阈值
     · 置信度过滤无法区分"这是两个不同的物体"还是"同一个物体的重复检测"
     · 只有比较框的位置重叠度（IoU）才能判断

  3. NMS 的核心逻辑: "同类别 + 高重叠 → 只保留分数最高的":
     · 按 final_score 排序，最高的先入选
     · 与已入选框 IoU 超过阈值（0.45）的同类别框 → 删除
     · 效果: 每个物体只保留一个"最准"的框

  4. IoU 阈值的选择:
     · 阈值太高 (如 0.9) → 很多重复框没被抑制 → 检测结果有冗余
     · 阈值太低 (如 0.1) → 靠得很近的不同物体被误删 → 漏检
     · 0.45 是常用值，在去重和保留之间取得平衡

  5. NMS 的局限性:
     · 密集场景（如一群人）中，相邻的不同人可能 IoU > 0.45
       → NMS 误删 → 漏检（这是 YOLO 系列的已知弱点）
     · 改进方案: Soft-NMS（降低分数而非直接删除）、
       DIoU-NMS（用 DIoU 替代 IoU）等
    """)


if __name__ == "__main__":
    main()
