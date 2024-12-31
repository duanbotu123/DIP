import open3d as o3d
import torch
import numpy as np

# 读取带色点云文件
point_cloud = o3d.io.read_point_cloud("point_cloud.ply")

# 将点云数据转换为 PyTorch 张量
points = torch.tensor(np.asarray(point_cloud.points), dtype=torch.float32)
colors = torch.tensor(np.asarray(point_cloud.colors), dtype=torch.float32)

# 定义旋转和平移参数
tx, ty, tz = 0.5, 1., 1.5 # 平移参数
# roll, pitch, yaw = torch.radians(torch.tensor(10.0)), torch.radians(torch.tensor(20.0)), torch.radians(torch.tensor(30.0))  # 旋转参数

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to a rotation matrix using PyTorch."""
    R_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(roll), -torch.sin(roll)],
                        [0, torch.sin(roll), torch.cos(roll)]])
    
    R_y = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                        [0, 1, 0],
                        [-torch.sin(pitch), 0, torch.cos(pitch)]])
    
    R_z = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                        [torch.sin(yaw), torch.cos(yaw), 0],
                        [0, 0, 1]])
    
    # Combined rotation matrix
    R = torch.mm(R_z, torch.mm(R_y, R_x))
    return R

# 计算旋转矩阵
# R = torch.tensor([[ 0.2916, -0.1497,  0.9447],
#          [ 0.9478,  0.1782, -0.2643],
#          [-0.1288,  0.9725,  0.1939]])
# print(torch.det(R))

random_rotation_matrix1 = np.random.rand(3, 3)
u1, _, vh1 = np.linalg.svd(random_rotation_matrix1)
Rx = torch.tensor(u1 @ vh1).float()
if torch.det(Rx)<0:
    Rx=-Rx
# Rx = torch.eye(3)

Tx = torch.rand(3)*1.5

# 应用旋转和平移变换
transformed_points = torch.mm(points, Rx.T) + Tx

# 随机删除一部分点
num_points = transformed_points.shape[0]
mask = torch.rand(num_points) > 0.50  # OK的点将被保留
transformed_points = transformed_points[mask]
transformed_colors = colors[mask]

# # 在一些点的颜色和位置上加扰动
noise_intensity = 0.01
transformed_points += noise_intensity * torch.randn_like(transformed_points)
transformed_colors += noise_intensity * torch.randn_like(transformed_colors)
transformed_colors = torch.clamp(transformed_colors, 0, 1)  # 确保颜色值在[0, 255]之间

# 将处理后的点云转换回 Open3D 格式
processed_point_cloud = o3d.geometry.PointCloud()
processed_point_cloud.points = o3d.utility.Vector3dVector(transformed_points.numpy())
processed_point_cloud.colors = o3d.utility.Vector3dVector(transformed_colors.numpy())

# 保存处理后的点云为新的 ply 文件
o3d.io.write_point_cloud("processed_pt.ply", processed_point_cloud)

# print("点云处理完成并已保存为 processed_pt.ply")
print('Transformation = \n', Rx, '\n', Tx)
