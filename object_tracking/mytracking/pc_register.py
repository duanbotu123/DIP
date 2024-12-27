import os
import torch
import numpy as np
from torch.optim import Adam
import torch.nn as nn
import open3d as o3d
import copy

def draw_registration_result(source, target, transformation, name):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    combined_point_cloud = source_temp + target_temp
    o3d.io.write_point_cloud(f"{name}.ply", combined_point_cloud)
    # o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: 使用大小为为{}的体素下采样点云.".format(voxel_size))
    pcd_down = pcd.voxel_down_sample(voxel_size)
 
    radius_normal = voxel_size * 2
    # print(":: 使用搜索半径为{}估计法线".format(radius_normal))
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
 
    radius_feature = voxel_size * 5
    # print(":: 使用搜索半径为{}计算FPFH特征".format(radius_feature))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source, target, voxel_size, save_ply=False):
    # print(":: 加载点云并转换点云的位姿.")
    # source = o3d.io.read_point_cloud("point_cloud.ply")
    # target = o3d.io.read_point_cloud("processed_pt.ply")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    if save_ply:
        draw_registration_result(source, target, np.identity(4), 'before')
 
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: 对下采样的点云进行RANSAC配准.")
    # print("   下采样体素的大小为： %.3f," % voxel_size)
    # print("   使用宽松的距离阈值： %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result
 
 


def find_transformation(pt1, pt2, path1=None, path2=None, save_ply=False):
    # source = o3d.io.read_point_cloud("point_cloud.ply")
    # target = o3d.io.read_point_cloud("processed_pt.ply")
    voxel_size = 0.05  # means 5cm for this dataset
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pt1, pt2, voxel_size, save_ply)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    matrix = torch.tensor(result_ransac.transformation, dtype=torch.float32)
    modify = torch.tensor([[0., 0, 1, 0], [1, 0, 0, 0], [0, 1 ,0, 0], [0, 0, 0, 1]])
    if save_ply:
        draw_registration_result(source_down, target_down, result_ransac.transformation, 'after')
        print('Ply files have been saved.')
    return matrix @ modify


if __name__== '__main__':
    source = o3d.io.read_point_cloud("point_cloud.ply")
    target = o3d.io.read_point_cloud("processed_pt.ply")
    matrix = find_transformation(source, target, save_ply=True)
    print(matrix)

    