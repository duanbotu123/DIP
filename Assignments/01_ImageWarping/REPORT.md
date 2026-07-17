# Image Warping — 作业1报告

## 项目概述

本作业实现了数字图像处理课程的 Assignment 1，包含两部分内容：

1. **基本几何变换**：围绕图像中心实现缩放、旋转、平移和水平翻转的复合变换
2. **点引导图像变形**：基于 Moving Least Squares (MLS) 实现图像变形

---

## 任务一：基本图像几何变换

### 实现方法

在 `run_global_transform.py` 的 `apply_transform` 函数中，构建仿射变换矩阵完成复合变换。

**变换顺序**（从左到右应用到源点）：
```
M = Flip × Translate × Rotate × Scale
```

**关键矩阵**：

| 变换 | 矩阵构造 |
|------|---------|
| 缩放（绕中心） | `S = T(c) × diag(s,s,1) × T(-c)` |
| 旋转（绕中心） | `R = T(c) × Rot(θ) × T(-c)` |
| 平移 | `T = [I \| (tx, ty)]` |
| 水平翻转（绕中心） | `F = T(c) × diag(-1,1,1) × T(-c)` |

其中 `c = (w/2, h/2)` 为图像中心。

### 运行方式

```bash
python run_global_transform.py
```

### 交互界面

Gradio 界面提供滑块和复选框实时控制：
- Scale: 0.1 – 2.0
- Rotation: -180° – 180°
- Translation: -300 – 300 px
- Flip Horizontal: 开关

---

## 任务二：点引导图像变形 (MLS)

### 实现方法

在 `run_point_transform.py` 中实现基于 Moving Least Squares (MLS) 的仿射变形。

**算法**（MLS Affine Deformation, Schaefer et al. 2006）：

对每个输出像素 v：
1. 计算权重 `w_i = 1 / |p_i - v|^2`
2. 加权质心 `p* = Σ(w_i p_i) / Σ(w_i)`，`q* = Σ(w_i q_i) / Σ(w_i)`
3. 中心化 `p̂_i = p_i - p*`，`q̂_i = q_i - q*`
4. 构造 `A = Σ w_i · p̂_i^T q̂_i`，`B = Σ w_i · p̂_i^T p̂_i`
5. `M = B⁻¹ A`
6. 源位置 `f(v) = M · (v - p*) + q*`

使用 `cv2.remap` 双线性插值完成重采样。

### 运行方式

```bash
python run_point_transform.py
```

### 工作流程

1. 上传图像
2. 奇数次点击：蓝色标记控制点（源位置）
3. 偶数次点击：红色标记目标点（目标位置）
4. 绿色箭头显示映射关系
5. 点击 Run Warping 执行变形
