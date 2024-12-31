import os
import torch
import numpy as np
from torch.optim import Adam
import torch.nn as nn
import open3d as o3d


'''Goal: Input the rgbs, cameras, masks. Get the 6 dof pose of the object.'''

'''read the inputs and get coarse gaussians of object'''

'''used colored ICP to find the pose'''
source = o3d.io.read_point_cloud("point_cloud.ply")
target = o3d.io.read_point_cloud("processed_pt.ply")
source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

# print(source, target)

initial_transformation = np.identity(4)
# initial_transformation = np.array([[ 0.2916, -0.1497,  0.9447, 0],
#          [ 0.9478,  0.1782, -0.2643, 0],
#          [-0.1288,  0.9725,  0.1939, 0],
#          [0, 0, 0, 1]])
max_correspondence_distance = 0.02
criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100)
estimation_method = o3d.pipelines.registration.TransformationEstimationForColoredICP()


result_icp = o3d.pipelines.registration.registration_colored_icp(
    source,
    target,
    max_correspondence_distance,
    initial_transformation,
    estimation_method,
    criteria
)

print(result_icp.transformation)
'''transform the pose into 6dof form'''

'''Functions: 6 DoF to transformation matrix'''
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

def pose_to_transformation_matrix(tx, ty, tz, roll, pitch, yaw):
    """Convert a 6 DoF pose to a transformation matrix using PyTorch."""
    # Translation vector
    t = torch.tensor([tx, ty, tz])
    
    # Rotation matrix from Euler angles
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    
    # Create the transformation matrix
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T
