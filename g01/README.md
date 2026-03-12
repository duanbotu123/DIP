# 作业 01 - 图像变形


## 环境要求

安装依赖：

```bash
python -m pip install -r requirements.txt
```

依赖列表见 `requirements.txt`。

## 评测

运行全局几何变换交互界面：

```bash
python run_global_transform.py
```

运行点引导形变交互界面：

```bash
python run_point_transform.py
```
。

## 结果

运行：

```bash
python generate_results.py
```

生成文件保存在 `results/`。

| 任务 | 结果文件 | 复现命令 |
| --- | --- | --- |
| 全局变换 | `results/global_input.png` | `python generate_results.py` |
| 全局变换 | `results/global_scale_1p4.png` | `python generate_results.py` |
| 全局变换 | `results/global_rotate_35.png` | `python generate_results.py` |
| 全局变换 | `results/global_translate.png` | `python generate_results.py` |
| 全局变换 | `results/global_flip.png` | `python generate_results.py` |
| 全局变换 | `results/global_combo.png` | `python generate_results.py` |
| 点引导形变 | `results/point_controls.png` | `python generate_results.py` |
| 点引导形变 | `results/point_warped.png` | `python generate_results.py` |
| 点引导形变 | `results/point_before_after.png` | `python generate_results.py` |

### 参考代码
我这次主要参考了以下论文与代码来源：

- 作业里给出的核心参考论文 ，本作业用RBF完成
  - RBF: Image Warping by Radial Basis Functions  
    https://www.sci.utah.edu/~gerig/CS6640-F2010/Project3/Arad-1995.pdf

- 工程/API参考文档  
  - OpenCV 几何变换文档  
    https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html  
  - Gradio 官方文档  
    https://www.gradio.app/

