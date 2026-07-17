# DIP with PyTorch — 作业2报告

## 项目概述

本作业完成了数字图像处理课程的 Assignment 2，包含两部分内容：

1. **Poisson Image Blending**：使用 PyTorch 实现泊松图像编辑
2. **Pix2Pix**：基于全卷积网络 (FCN) 实现图像到图像的翻译

---

## 任务一：Poisson Image Blending

### 实现方法

填充 `run_blending_gradio.py` 中两个缺失部分：

#### 1. 多边形掩码生成 (`create_mask_from_points`)

使用 OpenCV 的 `fillPoly` 将多边形点填充为二值掩码：
- 多边形内部：255
- 多边形外部：0

#### 2. 拉普拉斯损失 (`cal_laplacian_loss`)

使用 PyTorch 的 `conv2d` 计算拉普拉斯差异：

离散拉普拉斯核：
```
[0,  1, 0]
[1, -4, 1]
[0,  1, 0]
```

损失函数：
```
Loss = mean(||∇²blended - ∇²foreground||² × mask)
```

其中 `∇²` 为拉普拉斯算子。通过 `groups=3` 分别计算 RGB 三个通道的拉普拉斯。

#### 优化策略

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 1e-3 (前半段) → 1e-4 (后半段) |
| 迭代次数 | 10,000 |
| 混合初始值 | blended = 0.9 × bg + 0.1 × fg |

### 运行方式

```bash
python run_blending_gradio.py
```

### 数据

`data_poission/` 包含三组测试图像：
- `equation/`：公式合成
- `monolisa/`：蒙娜丽莎
- `water/`：水面

---

## 任务二：Pix2Pix 全卷积网络

### 网络架构

Encoder-Decoder 结构，5 层下采样 + 5 层上采样：

```
输入: 3 × 256 × 256
  ↓ Conv2D(3→8, k=4, s=2, p=1) + BN + ReLU    → 8 × 128 × 128
  ↓ Conv2D(8→16, k=4, s=2, p=1) + BN + ReLU    → 16 × 64 × 64
  ↓ Conv2D(16→32, k=4, s=2, p=1) + BN + ReLU   → 32 × 32 × 32
  ↓ Conv2D(32→64, k=4, s=2, p=1) + BN + ReLU   → 64 × 16 × 16
  ↓ Conv2D(64→128, k=4, s=2, p=1) + BN + ReLU  → 128 × 8 × 8
  ↓ ConvTranspose2D(128→64, k=4, s=2, p=1) + BN + ReLU → 64 × 16 × 16
  ↓ ConvTranspose2D(64→32, k=4, s=2, p=1) + BN + ReLU  → 32 × 32 × 32
  ↓ ConvTranspose2D(32→16, k=4, s=2, p=1) + BN + ReLU  → 16 × 64 × 64
  ↓ ConvTranspose2D(16→8, k=4, s=2, p=1) + BN + ReLU   → 8 × 128 × 128
  ↓ ConvTranspose2D(8→3, k=4, s=2, p=1) + Tanh        → 3 × 256 × 256
输出: 3 × 256 × 256 (范围 [-1, 1])
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 数据集 | Facades |
| 损失函数 | L1 Loss |
| 优化器 | Adam (lr=0.001, betas=(0.5, 0.999)) |
| 学习率调度 | StepLR, step=200, gamma=0.2 |
| Batch Size | 100 |
| Epochs | 800 |

### 运行方式

```bash
bash download_facades_dataset.sh
python train.py
```
