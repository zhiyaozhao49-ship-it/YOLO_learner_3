# YOLOv3 教学演示项目

一系列独立的 Python 脚本，逐步讲解 YOLOv3 目标检测算法的核心概念。每个脚本自包含、可直接运行，无需外部数据集或预训练权重。

## 环境要求

- Python 3.12（miniconda）
- PyTorch
- 无需 GPU（所有脚本均为前向推理 / 纯算法演示）

## 运行方式

```powershell
$env:PYTHONIOENCODING = "utf-8"
python <script>.py
```

> Windows 终端必须设置 `PYTHONIOENCODING="utf-8"`，否则中文输出会乱码。

## 脚本课程顺序

### 第一阶段：理解骨干网络

| # | 脚本 | 主题 |
|---|------|------|
| 1 | `darknet53_shape_demo.py` | Darknet-53 骨干网络逐层 shape 分析，观察 416×416 如何下采样为 52×52、26×26、13×13 |
| 2 | `residual_block_demo.py` | 残差块最小实验，理解 shortcut 连接为什么要求输入输出尺寸一致 |

### 第二阶段：理解检测头输出

| # | 脚本 | 主题 |
|---|------|------|
| 3 | `yolo_v3_output_shape_demo.py` | 三尺度输出形状模拟，拆解 255 = 3 × (4 + 1 + 80) 的来源 |
| 4 | `anchor_assignment_demo.py` | 9 个 anchor 按面积排序后分配到三个尺度，解释小 anchor 为什么配高分辨率层 |
| 5 | `softmax_vs_sigmoid_demo.py` | Softmax vs Sigmoid 对比实验，理解 YOLOv3 为什么用 Sigmoid 做多标签分类 |

### 第三阶段：理解推理流程

| # | 脚本 | 主题 |
|---|------|------|
| 6 | `feature_fusion_demo.py` | FPN 式特征融合实验，深层特征上采样 + 浅层特征拼接 |
| 7 | `yolo_v3_reshape_demo.py` | 输出张量 [B, 255, H, W] → [B, H, W, 3, 85] 维度变换 |
| 8 | `yolo_v3_bbox_decode_demo.py` | 边界框解码：sigmoid(tx)+cx、exp(tw)×pw 的数学含义 |
| 9 | `yolo_v3_target_assignment_demo.py` | 训练时 GT 如何分配到具体尺度 / anchor / 网格位置 |
| 10 | `yolo_v3_postprocess_demo.py` | 置信度过滤 + NMS 非极大值抑制后处理流水线 |

每个脚本独立运行，脚本间无 import 依赖。

## 知识点速查

| 概念 | 对应脚本 |
|------|---------|
| Darknet-53 结构、残差块 | #1, #2 |
| 255 通道含义、anchor 分配 | #3, #4 |
| Sigmoid 多标签分类 | #5 |
| 特征融合（FPN） | #6 |
| 维度变换 reshape | #7 |
| 边界框解码公式 | #8 |
| Target Assignment | #9 |
| NMS 后处理 | #10 |
