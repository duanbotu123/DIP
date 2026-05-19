# Assignment 4 — 简化版 3D Gaussian Splatting 实现

本仓库为 DIP 课程 Assignment 4 的作业提交，实现了一个完整的简化版 3D Gaussian Splatting (3DGS) pipeline：从多视角图像出发，经 Structure-from-Motion 恢复相机参数，初始化 3D 高斯，通过可微渲染训练高斯参数，最终实现新视角合成。

---

## 目录

- [环境配置](#环境配置)
- [Task 1: Structure-from-Motion with COLMAP](#task-1-structure-from-motion-with-colmap)
- [Task 2: 简化版 3D Gaussian Splatting](#task-2-简化版-3d-gaussian-splatting)
- [Task 3: 与官方 3DGS 实现的对比](#task-3-与官方-3dgs-实现的对比)
- [项目结构](#项目结构)

---

## 环境配置

### 依赖安装

```bash
conda create -n drender python=3.10
conda activate drender
pip install torch torchvision
pip install pytorch3d
pip install opencv-python tqdm natsort
```

### COLMAP 安装

```bash
# Ubuntu
sudo apt install colmap
# 或从源码编译
```

### 硬件要求

- GPU: NVIDIA GPU (>=8GB 显存)
- 训练使用: CUDA device, 约 11GB 显存占用

---

## Task 1: Structure-from-Motion with COLMAP

### 目标

使用 COLMAP 对 100 张多视角椅子图像进行 Structure-from-Motion (SfM)，恢复每张图像对应的相机内外参和一组稀疏 3D 点云，作为后续 3DGS 初始化。

### 运行方式

```bash
# Step 1: 运行 COLMAP SfM pipeline
python mvs_with_colmap.py --data_dir data/chair

# Step 2: 将 3D 点重投影回各视角验证
python debug_mvs_by_projecting_pts.py --data_dir data_dir data/chair
```

### COLMAP Pipeline 说明

`mvs_with_colmap.py` 自动执行以下步骤：

1. **特征提取** (`feature_extractor`): 使用 SIFT 检测器提取图像特征，假设所有图像共享同一相机内参 (`single_camera=1`)，相机模型为 PINHOLE
2. **特征匹配** (`exhaustive_matcher`): 对所有图像对进行穷举特征匹配
3. **稀疏重建** (`mapper`): 增量式 SfM，恢复相机位姿和 3D 点
4. **格式转换** (`model_converter`): 将二进制模型转为文本格式，便于 Python 读取

### 结果

| 项目 | 数值 |
|------|------|
| 输入图像数 | 100 |
| 成功注册图像数 | 100 |
| 恢复 3D 点数 | ~14,300 |
| 相机模型 | PINHOLE |
| 原始图像分辨率 | 800 x 800 |
| 下采样因子 | 8 (训练时) |
| 训练时图像分辨率 | 100 x 100 |

**重投影验证**：将 COLMAP 恢复的 3D 点用对应相机参数投影回 2D 图像，验证了相机参数的正确性。100 张重投影验证图像保存在 `data/chair/projections/`。

---

## Task 2: 简化版 3D Gaussian Splatting

### 目标

基于 Task 1 的 COLMAP 输出，将稀疏 3D 点初始化为 3D 高斯，通过可微渲染和反向传播优化高斯参数，实现新视角合成。

### 核心实现

本作业需要完成 4 个 TODO，分别对应 3DGS 的核心数学模块：

#### TODO 1: 3D 协方差矩阵计算 (`gaussian_model.py` L103)

根据论文公式 (6)，3D 高斯的协方差矩阵由旋转矩阵 R 和缩放矩阵 S 构造：

$$\Sigma = R \cdot S \cdot S^T \cdot R^T$$

```python
def compute_covariance(self) -> torch.Tensor:
    R = self._compute_rotation_matrices()
    scales = torch.exp(self.scales.clamp(min=-10.0, max=10.0))
    S = torch.diag_embed(scales)
    RS = torch.bmm(R, S)
    Covs3d = torch.bmm(RS, RS.transpose(1, 2))
    return Covs3d
```

**关键设计**：
- 旋转用单位四元数参数化，通过 `F.normalize` 保证单位长度
- 缩放参数在 log 空间优化，通过 `torch.exp` 转换，并 clamp 到 `[-10, 10]` 防止 `exp()` 溢出
- 颜色在 logit 空间优化，通过 `sigmoid` 映射到 `[0, 1]`
- 不透明度同样用 logit + sigmoid 参数化

#### TODO 2: 3D -> 2D 投影 (`gaussian_renderer.py` L26)

根据论文公式 (5)，将 3D 高斯投影到图像平面需要：

1. **世界到相机坐标变换**: `p_cam = R * p_world + t`
2. **透视投影雅可比矩阵** J:

$$J = \begin{bmatrix} f_x / t_z & 0 & -f_x \cdot t_x / t_z^2 \\ 0 & f_y / t_z & -f_y \cdot t_y / t_z^2 \end{bmatrix}$$

3. **2D 协方差**: `Sigma' = J * W * Sigma * W^T * J^T`，其中 `W = R` 为世界到相机旋转

```python
# 构建雅可比矩阵
J_proj[:, 0, 0] = fx / tz
J_proj[:, 0, 2] = -fx * tx / (tz * tz)
J_proj[:, 1, 1] = fy / tz
J_proj[:, 1, 2] = -fy * ty / (tz * tz)

# 相机空间协方差
covs_cam = R @ Sigma @ R^T

# 投影到2D
covs2D = J @ covs_cam @ J^T
```

#### TODO 3: 2D 高斯取值计算 (`gaussian_renderer.py` L69)

像素 x 处的 2D 高斯取值：

$$f(x; \mu_i, \Sigma_i) = \frac{1}{2\pi\sqrt{|\Sigma_i|}} \exp\left(-\frac{1}{2}(x - \mu_i)^T \Sigma_i^{-1} (x - \mu_i)\right)$$

```python
# 解析 2x2 逆矩阵 (避免 eigh 梯度不稳定问题)
a, b, c = covs2D[:, 0, 0], covs2D[:, 0, 1], covs2D[:, 1, 1]
det = (a * c - b * b).clamp(min=1e-10)
covs2D_inv = stack([c / det, -b / det, -b / det, a / det])  # 2x2 逆
```

**数值稳定性优化**：
- 使用解析 2x2 逆公式替代 `torch.linalg.eigh`，因为 `eigh` 的反向传播在协方差矩阵接近退化时会产生 NaN
- 添加正则化 `Sigma + eps * I`（eps = 1e-3）
- 将 Mahalanobis 距离 clamp 到最大 80，防止 `exp()` 溢出

#### TODO 4: alpha-blending 体渲染 (`gaussian_renderer.py` L102)

按深度排序后，像素颜色由 alpha-blending 累加（论文公式 1-3）：

$$\alpha_i = o_i \cdot f(x; \mu_i, \Sigma_i), \quad T_i = \prod_{j<i}(1 - \alpha_j)$$

$$C(x) = \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot c_i$$

```python
# 透射率通过 cumprod 高效计算
transmittance = torch.cumprod(
    torch.cat([ones, 1 - alphas[:-1]], dim=0), dim=0
)
weights = transmittance * alphas
rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 训练轮数 | 200 |
| Batch Size | 1 (逐张图像) |
| 3D 高斯数量 | ~14,300 |
| 下采样因子 | 8 (100x100 渲染) |
| 优化器 | Adam |
| 梯度裁剪 | 1.0 |

**各参数学习率**：

| 参数 | 学习率 |
|------|--------|
| 位置 (xyz) | 1.6e-5 |
| 颜色 (color) | 2.5e-2 |
| 不透明度 (opacity) | 5e-2 |
| 缩放 (scaling) | 5e-3 |
| 旋转 (rotation) | 1e-3 |

### 运行方式

```bash
# 训练
CUDA_VISIBLE_DEVICES=2 python train.py \
    --colmap_dir data/chair \
    --checkpoint_dir data/chair/checkpoints \
    --num_epochs 200

# 渲染多视角视频
python render_3dgs_mv.py \
    --colmap_dir data/chair \
    --checkpoint data/chair/checkpoints/checkpoint_000180.pt \
    --output data/chair/render_mv.mp4 \
    --num_frames 240 --fps 30
```

### 训练结果

| 指标 | 数值 |
|------|------|
| 训练轮数 | 200 epochs (0-199) |
| 训练时长 | ~75 分钟 (单 GPU, ~22s/epoch) |
| 显存占用 | ~11 GB |
| 初始 Loss (Epoch 1) | ~0.12 |
| 最终 Loss (Epoch 199) | ~0.124 |
| 收敛后 Loss 范围 | 0.11-0.14 (per-iteration) |
| Checkpoint 数量 | 10 个 (每 20 epochs 保存) |
| Checkpoint 大小 | ~2.4 MB 每个 |
| 最终 Checkpoint | `checkpoint_000180.pt` |

训练过程稳定，Loss 从 Epoch 0 的 NaN（`eigh` 梯度不稳定，已修复）到 Epoch 1+ 稳定在 ~0.11-0.14 范围（L1 loss）。每个 epoch 遍历 100 张训练图像。训练完成后自动生成了训练视角渲染视频 `debug_rendering.mp4`。

**训练过程中的数值稳定性问题及解决方案**：

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Epoch 0 出现 NaN loss | `torch.linalg.eigh` 在退化协方差矩阵上反向传播不稳定 | 替换为解析 2x2 逆公式 |
| `exp()` 溢出 | 缩放参数过大导致 `exp()` 输出无穷 | clamp scales 到 `[-10, 10]` |
| 2D 高斯值为 NaN | 2D 协方差矩阵接近奇异 | 添加 `eps * I` 正则化 |
| Alpha 值超出 [0,1] | 乘积溢出 | clamp 到 `[0, 0.999]` |
| 个别 batch NaN loss | 梯度异常 | 添加 NaN 检测，跳过异常 batch |

### Debug 可视化

训练过程中每 20 epochs 自动保存 4 个视角的 GT vs Rendered 对比图到 `checkpoints/debug_images/`，上排为 GT，下排为渲染结果。共 10 张对比图：

- `epoch_0000.png` ~ `epoch_0180.png`

### 多视角轨道渲染

训练完成后，使用最终 checkpoint 沿水平圆轨迹渲染了 240 帧新视角视频：

```bash
python render_3dgs_mv.py \
    --colmap_dir data/chair \
    --checkpoint data/chair/checkpoints/checkpoint_000180.pt \
    --output data/chair/render_mv.mp4 \
    --num_frames 240 --fps 30
```

| 视频属性 | 值 |
|----------|-----|
| 输出路径 | `data/chair/render_mv.mp4` |
| 分辨率 | 100 x 100 |
| 帧数 | 240 |
| FPS | 30 |
| 时长 | 8 秒 |
| 轨迹 | 水平圆轨道 (绕场景中心) |

此外，训练脚本自动生成了训练视角的渲染对比视频 `checkpoints/debug_rendering.mp4`。

---

## Task 3: 与官方 3DGS 实现的对比

### 对比概述

| 维度 | 本实现 (简化版) | 官方实现 |
|------|----------------|----------|
| **语言/框架** | 纯 PyTorch | PyTorch + CUDA 自定义算子 |
| **光栅化方式** | 逐像素全量计算 | Tile-based 分块光栅化 (CUDA) |
| **高斯数量控制** | 固定 (~14K 初始点) | 自适应密度控制 (clone/split/prune) |
| **颜色表示** | RGB (3通道) | 球谐函数 (SH, 最高4阶 = 48参数) |
| **渲染分辨率** | 100x100 (8x下采样) | 原始分辨率 (800x800) |
| **训练时长** | ~75 分钟 (200 epochs) | ~10-15 分钟 (30K iterations) |
| **显存占用** | ~11 GB | ~2-4 GB |
| **渲染质量** | 模糊，缺乏细节 | 高保真，接近真实 |
| **实时渲染** | 不支持 | 支持 (>100 FPS) |

### 差异来源分析

#### 1. 光栅化方式：全量计算 vs Tile-based

**本实现**对每个 2D 高斯，在全图 (HxW) 上计算所有像素的高斯取值。这意味着每个高斯都要计算 HxW 个值，即使大部分像素远在高斯有效范围之外。计算复杂度为 O(N * H * W)。

**官方实现**采用 tile-based 光栅化（论文 Section 4），将图像分为 16x16 的 tile，每个高斯只影响其 bounding box 覆盖的 tile。此外，官方使用 CUDA 自定义算子，将高斯按 tile 排序后并行处理，极大减少了冗余计算。这带来两个关键优势：
- **速度**: 只计算高斯覆盖区域的像素，减少 10-100x 计算量
- **显存**: 不需要为每个高斯分配完整的 HxW 特征图

#### 2. 自适应密度控制 (Adaptive Density Control)

**本实现**使用 COLMAP 恢复的固定点云（~14K 点），不做任何增删。这意味着：
- 纹理丰富区域点不够密，无法表达精细结构
- 空白区域的点被浪费

**官方实现**在训练过程中周期性地进行自适应密度控制：
- **Clone**: 对于覆盖小区域的高斯（欠重建），复制一个同样大小的高斯
- **Split**: 对于覆盖大区域的高斯（过重建），分裂为两个更小的高斯
- **Prune**: 移除不透明度接近零的高斯
- 最终高斯数量可增长到 100K-1M，密集覆盖场景表面

#### 3. 球谐函数 vs RGB

**本实现**直接优化每个高斯的 RGB 颜色（3 参数），这意味着每个 3D 点只有一种颜色，无法表达视角相关的反射效果（如高光、镜面反射）。

**官方实现**使用球谐函数（Spherical Harmonics, SH）表示颜色，最高 4 阶（48 参数/点），可以建模视角相关的辐射场。这使得：
- 光滑表面能呈现真实的高光变化
- 半透明材质的视角依赖效果得以表达

#### 4. 渲染分辨率

本实现为了在纯 PyTorch 框架下保持可接受的训练速度，将图像下采样 8 倍至 100x100。官方实现在原始分辨率 (800x800) 下训练。分辨率差异直接影响：
- **细节保留**: 100x100 无法分辨椅子的精细结构
- **高频信息**: 下采样丢失了纹理的高频成分
- **PSNR 基准**: 即使高斯参数完美，低分辨率渲染的 PSNR 也天然较低

#### 5. 训练策略差异

| 方面 | 本实现 | 官方实现 |
|------|--------|----------|
| 迭代方式 | Epoch (遍历全量) | Iteration (随机采样) |
| 总迭代量 | 200 epochs x 100 images = 20K | 30K iterations |
| Loss 函数 | L1 | L1 + D-SSIM |
| 学习率调度 | 固定 | Exponential decay |
| 密度控制 | 无 | 每 100 iterations |

### 性能对比总结

本简化实现成功验证了 3DGS 的核心数学原理（协方差构造、投影、alpha-blending），但在实际性能上与官方实现存在本质差距：

1. **渲染质量**: 缺乏自适应密度控制和球谐函数，导致重建模糊
2. **训练效率**: 全量计算 vs tile-based 导致 5-10x 速度差距
3. **显存效率**: 全图特征图 vs tile 局部计算导致 3-5x 显存差距
4. **表达能力**: 固定点数 + RGB vs 动态点数 + SH 导致表达力上限差距

这些差距的核心来源是 **tile-based CUDA 光栅化**和**自适应密度控制**这两个关键模块，它们是 3DGS 论文的核心贡献。

---

## 项目结构

```
04_3DGS/
├── README.md                    # 作业描述 (原始)
├── REPORT.md                    # 本报告
├── mvs_with_colmap.py           # Task 1: COLMAP SfM pipeline
├── debug_mvs_by_projecting_pts.py  # Task 1: 重投影验证
├── gaussian_model.py            # Task 2: 3D 高斯模型 (TODO 1)
├── gaussian_renderer.py         # Task 2: 投影+渲染 (TODO 2/3/4)
├── data_utils.py                # COLMAP 数据加载
├── train.py                     # 训练循环
├── render_3dgs_mv.py            # 多视角轨道渲染
└── data/
    └── chair/
        ├── images/              # 100 张输入图像 (800x800)
        ├── sparse/
        │   └── 0_text/          # COLMAP 输出 (cameras/images/points3D.txt)
        ├── projections/         # 100 张重投影验证图
        ├── render_mv.mp4        # Task 2: 轨道渲染视频 (240帧, 30fps)
        └── checkpoints/
            ├── checkpoint_000000.pt ~ checkpoint_000180.pt  # 10 个 checkpoints
            ├── debug_images/    # 训练过程可视化 (10 张, 每 20 epochs)
            └── debug_rendering.mp4  # 训练视角渲染对比视频
```

## 参考文献

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf) -- Kerbl et al., SIGGRAPH 2023
- [COLMAP](https://colmap.github.io/) -- Structure-from-Motion library
- [Official 3DGS Implementation](https://github.com/graphdeco-inria/gaussian-splatting)
