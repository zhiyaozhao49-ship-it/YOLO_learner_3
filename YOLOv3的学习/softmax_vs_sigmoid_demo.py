"""
Softmax vs Sigmoid 类别预测对比实验
====================================

功能：用具体的数值例子对比 Softmax 和 Sigmoid 在分类任务中的行为差异，
      解释为什么 YOLOv3 选择 Sigmoid（多标签独立分类）而非 Softmax。

核心区别（一句话）:
    · Softmax: 所有类别的概率之和 = 1，互相竞争（"只能选一个"）
    · Sigmoid: 每个类别独立输出 0~1，互不影响（"可以选多个"）

参考：YOLOv3 论文 —— "An Incremental Improvement" (2018)
"""

import torch
import torch.nn.functional as F


def main():
    print("=" * 70)
    print("  Softmax vs Sigmoid 类别预测对比")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # 实验 1: 基础向量对比
    # ═══════════════════════════════════════════════════════════════════════

    logits = torch.tensor([2.0, 1.5, -1.0])

    softmax_result = F.softmax(logits, dim=0)
    sigmoid_result = torch.sigmoid(logits)

    print(f"\n  [实验 1] logits = {logits.tolist()}")
    print()
    print(f"  {'类别':>6s}  {'logits':>8s}  {'Softmax':>10s}  {'Sigmoid':>10s}")
    print("  " + "-" * 42)
    for i in range(len(logits)):
        print(f"  {i:>6d}  {logits[i]:>8.2f}  "
              f"{softmax_result[i]:>10.6f}  {sigmoid_result[i]:>10.6f}")
    print("  " + "-" * 42)
    print(f"  {'求和':>6s}  {'':>8s}  "
          f"{softmax_result.sum():>10.6f}  {sigmoid_result.sum():>10.6f}")

    print(f"""
  观察结果:
    · Softmax 求和 = {softmax_result.sum():.6f}  （恒等于 1.0，互斥分配）
    · Sigmoid 求和 = {sigmoid_result.sum():.6f}  （可以 > 1，各自独立）

    · Softmax 中类别 0 概率高 → 类别 1、2 就被"挤压"变低
    · Sigmoid 中类别 0 概率高 → 不影响类别 1、2 的判断
    """)

    # ═══════════════════════════════════════════════════════════════════════
    # 实验 2: 多标签场景 —— 一个女人坐在自行车上
    # ═══════════════════════════════════════════════════════════════════════
    #
    # 假设只有 3 个类别: [人, 自行车, 狗]
    # 场景: 一张图片里同时有"人"和"自行车"（但没有"狗"）
    # 理想的预测: 人=高, 自行车=高, 狗=低

    labels = ["人", "自行车", "狗"]
    # 网络对三个类别的原始打分（logits）
    # "人"和"自行车"的得分都很高，"狗"的得分低
    multi_logits = torch.tensor([3.0, 2.5, -2.0])

    sm = F.softmax(multi_logits, dim=0)
    sg = torch.sigmoid(multi_logits)

    print("  [实验 2] 多标签场景: 图片里同时有「人」和「自行车」")
    print()
    print(f"  {'类别':>8s}  {'logits':>8s}  {'Softmax':>10s}  {'Sigmoid':>10s}  "
          f"{'Softmax判断':>12s}  {'Sigmoid判断':>12s}")
    print("  " + "-" * 72)
    for i in range(3):
        sm_judge = "✓ 有物体" if sm[i] > 0.5 else "✗ 无物体"
        sg_judge = "✓ 有物体" if sg[i] > 0.5 else "✗ 无物体"
        print(f"  {labels[i]:>8s}  {multi_logits[i]:>8.2f}  "
              f"{sm[i]:>10.6f}  {sg[i]:>10.6f}  "
              f"{sm_judge:>12s}  {sg_judge:>12s}")
    print("  " + "-" * 72)
    print(f"  {'求和':>8s}  {'':>8s}  {sm.sum():>10.6f}  {sg.sum():>10.6f}")

    print(f"""
  问题暴露:
    · Softmax 求和 = 1.0，概率被"抢"了
      「人」得分 3.0，Softmax 给了 {sm[0]:.4f}
      「自行车」得分 2.5，Softmax 只给了 {sm[1]:.4f}
      明明两个都在图片里，但 Softmax 让它们互相竞争

    · Sigmoid 各自独立判断
      「人」sigmoid(3.0) = {sg[0]:.4f} → 高概率 ✓
      「自行车」sigmoid(2.5) = {sg[1]:.4f} → 高概率 ✓
      两个都正确地报出了高概率，互不干扰

    · 如果用 Softmax 做多标签:
      只有 1 个类别能 > 0.5（因为和 = 1.0）
      永远无法同时输出"这个框里有人 AND 有自行车"
    """)

    # ═══════════════════════════════════════════════════════════════════════
    # 实验 3: 改变一个 logit 观察"牵连效应"
    # ═══════════════════════════════════════════════════════════════════════

    print("  [实验 3] 牵连效应: 固定其他值不变，只增大类别 0 的 logit")
    print()

    base_logits = torch.tensor([1.0, 1.0, 1.0])
    print(f"  {'类别0 logit':>12s}  {'Softmax[1]':>12s}  {'Softmax[2]':>12s}  "
          f"{'Sigmoid[1]':>12s}  {'Sigmoid[2]':>12s}")
    print("  " + "-" * 68)

    for val in [1.0, 3.0, 5.0, 10.0]:
        test = torch.tensor([val, 1.0, 1.0])
        sm_t = F.softmax(test, dim=0)
        sg_t = torch.sigmoid(test)
        print(f"  {val:>12.1f}  {sm_t[1]:>12.6f}  {sm_t[2]:>12.6f}  "
              f"{sg_t[1]:>12.6f}  {sg_t[2]:>12.6f}")

    print(f"""
  观察:
    · Softmax: 类别 0 的 logit 增大 → 类别 1、2 的概率被"挤压"到接近 0
      这就是"竞争"——一个类别的增益 = 其他类别的损失

    · Sigmoid: 类别 0 的 logit 增大 → 类别 1、2 完全不受影响
      这就是"独立"——每个类别只看自己的 logit

    结论:
      · Softmax 适合"互斥分类"（一张图片只能是猫 OR 狗 OR 鸟）
      · Sigmoid 适合"多标签分类"（一张图片可以同时有猫 AND 狗 AND 鸟）
    """)

    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  为什么 YOLOv3 使用 Sigmoid（逻辑分类器）？")
    print("=" * 70)
    print("""
  1. COCO 数据集中存在多标签场景:
     · 一个人骑着自行车 → 标签同时包含「人」和「自行车」
     · 一辆车旁边有只狗 → 标签同时包含「汽车」和「狗」
     · 如果用 Softmax，一个 anchor 只能预测一个类别，无法覆盖

  2. YOLOv3 的检测粒度是 anchor 级别:
     · 每个 anchor 独立预测 80 个类别的概率
     · 某个 anchor 覆盖了"人骑自行车"的区域
       → 应该同时报出「人」和「自行车」两个类别
     · Sigmoid 让 80 个类别各自独立判断，不互相排斥

  3. 对比 YOLOv2:
     · YOLOv2 使用 Softmax 做类别分类
     · 论文中指出: Softmax 假设类别互斥，但 COCO 中存在重叠
       (如「女人」和「人」不是互斥的)
     · YOLOv3 改用 Sigmoid，放弃了互斥假设，更灵活

  4. 数学等价性:
     · 当只有一个正类时，Sigmoid 和 Softmax 的训练效果接近
     · 当有多个正类时，Softmax 会"分走"概率，导致漏检
     · Sigmoid 没有这个问题，每个类别独立优化

  总结:
      Softmax  → 「选出一个冠军」→ 单标签分类（ImageNet 分类）
      Sigmoid  → 「每个都独立打分」→ 多标签分类（YOLOv3 检测）
    """)


if __name__ == "__main__":
    main()
