# test_render.py
import pyrender
import numpy as np
import trimesh
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# 创建立方体网格
cube_mesh = trimesh.creation.box()
# 转换为PyRender网格
mesh = pyrender.Mesh.from_trimesh(cube_mesh)
# 创建简单立方体

scene = pyrender.Scene()
scene.add(mesh)

# 渲染测试
renderer = pyrender.OffscreenRenderer(800, 600)
color, depth = renderer.render(scene)
print("渲染成功!")