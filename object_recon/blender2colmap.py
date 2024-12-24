import bpy
import os
import math
import json
import sys
import mathutils
from mathutils import Matrix
import numpy as np
import bmesh


def setup_output_dir(output_dir):
    # 创建 images 和 sparse 文件夹
    images_dir = os.path.join(output_dir, 'images')
    sparse_dir = os.path.join(output_dir, 'sparse/0')
    
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(sparse_dir):
        os.makedirs(sparse_dir)

    return images_dir, sparse_dir

def import_mesh(mesh_path):
    # 删除所有物体
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # 导入 OBJ 文件
    bpy.ops.wm.obj_import(filepath=mesh_path)
    
# 计算包围球半径
def calculate_bounding_sphere_radius(mesh_obj):
    # 获取mesh数据
    mesh = mesh_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    # 找到所有顶点的最小和最大坐标
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    for v in bm.verts:
        x, y, z = v.co
        min_x, min_y, min_z = min(min_x, x), min(min_y, y), min(min_z, z)
        max_x, max_y, max_z = max(max_x, x), max(max_y, y), max(max_z, z)
    
    # 计算包围球半径
    radius = math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2 + (max_z - min_z)**2) / 2
    return radius

def normalize_mesh(obj, target_size=1.0):
    # 确保对象处于活动对象
    bpy.context.view_layer.objects.active = obj

    # 计算物体的边界框 (bounding box)
    # 获取物体在世界坐标中的位置
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

    for vertex in obj.data.vertices:
        world_coords = obj.matrix_world @ vertex.co
        min_x = min(min_x, world_coords.x)
        min_y = min(min_y, world_coords.y)
        min_z = min(min_z, world_coords.z)
        max_x = max(max_x, world_coords.x)
        max_y = max(max_y, world_coords.y)
        max_z = max(max_z, world_coords.z)

    # 获取物体的边界框尺寸
    bbox_size = math.sqrt(
        (max_x - min_x)**2 +
        (max_y - min_y)**2 +
        (max_z - min_z)**2
    )

    # 计算缩放因子，保证物体的最大尺寸为target_size
    scale_factor = target_size / bbox_size if bbox_size != 0 else 1

    print(f"scale_factor: {scale_factor}")

    # 应用缩放
    obj.scale = (scale_factor, scale_factor, scale_factor)

    # 更新物体的变换矩阵
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)

    return scale_factor


def create_camera_views(obj, num_views, radius=10):
    # 创建相机围绕 Mesh 生成多个视角
    cameras = []
    for i in range(num_views):
        # 创建一个相机
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = f"Camera_{i}"
        # 计算相机位置，假设相机沿球体表面均匀分布
        theta = math.acos(2 * i / num_views - 1)  # 仰角
        phi = math.pi * (1 + 5**0.5) * i  # 方位角（黄金角螺旋）

        # 球坐标到笛卡尔坐标的转换
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)

        camera.location = (x, y, z)

        # 创建TrackTo约束，确保相机始终朝向目标物体
        track_to = camera.constraints.new(type='TRACK_TO')
        track_to.target = obj
        track_to.track_axis = 'TRACK_NEGATIVE_Z'  # 相机的Z轴朝向目标
        track_to.up_axis = 'UP_Y'  # Y轴为向上的方向

        # 保存相机
        cameras.append(camera)
    
    return cameras


def render_images(cameras, images_dir, output_file_prefix):
    # 渲染图像并保存
    image_files = []
    for i, camera in enumerate(cameras):
        # 设置渲染相机
        bpy.context.scene.camera = camera
        
        # 设置渲染输出路径
        image_path = os.path.join(images_dir, f"{output_file_prefix}_{i:03d}.png")
        bpy.context.scene.render.filepath = image_path

        # 执行渲染
        bpy.ops.render.render(write_still=True)
        image_files.append(image_path)
    
    return image_files

def save_camera_parameters(cameras, images, output_dir):
    cameras_txt_path = os.path.join(output_dir, 'sparse/0/cameras.txt')
    images_txt_path = os.path.join(output_dir, 'sparse/0/images.txt')
    
    # 写入 cameras.txt
    with open(cameras_txt_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        
        width = bpy.context.scene.render.resolution_x
        height = bpy.context.scene.render.resolution_y
        camera = bpy.context.scene.camera.data
        fx = fy = camera.lens  # 焦距
        # 转化成像素单位
        #fx = fx * width / camera.sensor_width
        #fy = fy * height / camera.sensor_height
        fx = fy = 2000
        cx = width / 2    # 主点
        cy = height / 2
        
        # 使用空格分隔，PINHOLE模型
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    # 写入 images.txt
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n\n")
        
        for i, (camera, image_path) in enumerate(zip(cameras, images)):
            # 获取相机到世界的变换矩阵
            world_matrix = camera.matrix_world.copy()


            for j in range(3):
                world_matrix[j][1] *= -1
                world_matrix[j][2] *= -1

            # 绕x轴旋转90°
            world_matrix = Matrix.Rotation(math.radians(-90), 4, 'X') @ world_matrix

            c2w = world_matrix.inverted()
            R = c2w.to_quaternion().to_matrix()
            T = c2w.translation
            #breakpoint()
            # 从旋转矩阵计算四元数
            rot = R.to_quaternion()
            qw, qx, qy, qz = rot.w, rot.x, rot.y, rot.z
            #breakpoint()
            
            # 获取图像文件名
            image_name = os.path.basename(image_path)
            
            # 写入到文件
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {T.x} {T.y} {T.z} 1 {image_name}\n\n")




def save_3d_points(obj_path, sparse_dir, scale_factor=1.0):
    import cv2
    import numpy as np
    import os

    points3d_path = os.path.join(sparse_dir, 'points3D.txt')
    
    # 读取顶点和UV数据
    vertices = []  # 存储v
    uvs = []      # 存储vt
    faces = []    # 存储f
    texture_file = None
    
    # 读取MTL文件以获取纹理路径
    def parse_mtl(mtl_path):
        with open(mtl_path, 'r') as f:
            for line in f:
                if line.startswith('map_Kd'):
                    # 获取颜色纹理路径
                    return line.split()[1]
        return None

    # 读取OBJ文件
    #breakpoint()
    obj_dir = os.path.dirname(obj_path)
    with open(obj_path, 'r') as f:
        mtl_file = None
        for line in f:
            if line.startswith('mtllib'):
                # 读取MTL文件
                mtl_file = line.split()[1]
                mtl_path = os.path.join(obj_dir, mtl_file)
                texture_file = parse_mtl(mtl_path)
                if texture_file:
                    texture_file = os.path.join(obj_dir, texture_file)
            elif line.startswith('v '):
                # 顶点坐标
                v = [float(x) for x in line[2:].split()]
                vertices.append(v)
            elif line.startswith('vt '):
                # UV坐标
                vt = [float(x) for x in line[3:].split()]
                uvs.append(vt)
            elif line.startswith('f '):
                # 面信息 (顶点索引/UV索引/法线索引)
                f = [[int(i.split('/')[0])-1, int(i.split('/')[1])-1] for i in line[2:].split()]
                faces.append(f)

    # 读取纹理图像
    image = cv2.imread(texture_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB
    height, width = image.shape[:2]

    # 创建顶点颜色映射
    vertex_colors = {}  # 存储每个顶点的颜色
    
    # 为每个顶点获取颜色
    for face in faces:
        for vertex_idx, uv_idx in face:
            if vertex_idx not in vertex_colors:
                # 获取UV坐标
                u, v = uvs[uv_idx]
                # 转换UV到图像坐标
                x = int(u * (width - 1))
                y = int((1 - v) * (height - 1))
                # 获取颜色
                color = image[y, x]
                vertex_colors[vertex_idx] = color
    # 写入points3D.txt
    with open(points3d_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
        
        for i, vertex in enumerate(vertices):
            x, y, z = [coord * scale_factor for coord in vertex]
            r, g, b = vertex_colors.get(i, (255, 255, 255))
            error = 0
            f.write(f"{i} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error}\n")

def setup_scene():
    # 设置基本渲染属性
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    
    # 设置世界背景为黑色
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    # 设置基本环境
    world.use_nodes = True
    node_tree = world.node_tree
    node_tree.nodes.clear()
    
    # 创建背景节点
    background = node_tree.nodes.new('ShaderNodeBackground')
    background.inputs['Color'].default_value = (0, 0, 0, 1)  # 黑色背景
    background.inputs['Strength'].default_value = 0.0  # 完全关闭环境光
    
    # 创建输出节点
    output = node_tree.nodes.new('ShaderNodeOutputWorld')
    node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])
    
    # 禁用所有阴影和后处理效果
    bpy.context.scene.eevee.use_shadows = False
    bpy.context.scene.eevee.use_soft_shadows = False
    bpy.context.scene.eevee.use_gtao = False
    bpy.context.scene.eevee.use_bloom = False
    bpy.context.scene.eevee.use_ssr = False
    
    # 添加多个点光源以获得均匀照明
    light_positions = [
        (5, 5, 5),
        (-5, 5, 5),
        (5, -5, 5),
        (-5, -5, 5),
        (0, 0, 10),
        (0, 0, -5)
    ]
    
    for pos in light_positions:
        bpy.ops.object.light_add(type='POINT', location=pos)
        light = bpy.context.active_object
        light.data.energy = 500  # 降低每个光源的强度
        light.data.use_shadow = False  # 禁用阴影
    
    # 其他渲染设置
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.view_settings.exposure = 0.0


def setup_lighting():
    # 创建一个新的点光源
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 5))
    light = bpy.context.active_object
    light.data.energy = 1000  # 增加光源强度
    
    # 可以添加多个光源
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
    sun = bpy.context.active_object
    sun.data.energy = 5  # 太阳光强度

def main():
    # 获取命令行参数（输出路径和 mesh 路径）
    if len(sys.argv) < 4:
        print("Usage: blender --background --python script.py -- <output_dir> <mesh_file>")
        sys.exit(1)

    output_dir = sys.argv[sys.argv.index('--') + 1]
    mesh_file = sys.argv[sys.argv.index('--') + 2]
    
    # 设置输出路径并创建文件夹
    images_dir, sparse_dir = setup_output_dir(output_dir)

    # 设置渲染分辨率
    bpy.context.scene.render.resolution_x = 1600  # 设置宽度
    bpy.context.scene.render.resolution_y = 900    # 设置高度

    # 导入 Mesh
    import_mesh(mesh_file)

    # 获取导入的物体
    obj = bpy.context.selected_objects[0]
    if obj is not None:
        scale_factor = normalize_mesh(obj, target_size=5.0)  # 调整mesh大小

    #计算合适的半径
    radius = calculate_bounding_sphere_radius(obj) * 5

    # 设置黑色背景
    setup_scene()

    # 设置光照
    #setup_lighting()

    # 创建相机视角（这里假设生成 36 个视角，可以根据需求调整）
    cameras = create_camera_views(obj, num_views=100, radius=radius)

    # 渲染图像并保存
    output_file_prefix = os.path.splitext(os.path.basename(mesh_file))[0]
    images = render_images(cameras, images_dir, output_file_prefix)

    # 保存相机参数和图像参数
    save_camera_parameters(cameras, images, output_dir)

    # 保存 Mesh 顶点数据
    save_3d_points(mesh_file, sparse_dir, scale_factor)

    print("COLMAP 数据集已生成！")

if __name__ == "__main__":
    main()
