# origin repo [AnimatableGaussians](https://github.com/lizhe00/AnimatableGaussians.git)


# run

按照原仓库配环境和下载smplx

angs要求的格式和easymocap的输出不同

- 相机格式转换
  - utils/transfer_camera_angs.py
  
    如果有相机没用上，需要手动改下names里的相机名
    - extri: 外参路径
    - intri: 内参路径
    - save_path: 保存路径
    - 4k/1080p

- smpl坐标系转换
  - utils/standard_smpl

    easymocap输出的smpl坐标系和angs不同，要转换

    - raw_folder: easymocap输出参数
    - joint_folder: easymocap输出joint
    - out_folder: 保存路径

- 写成angs需要的npz文件
  - utils/mocap2npz.py

    - json_dir: smpl序列
    - output_file: 保存位置
  
- angs的图片命名是八位数字，填充0，要改图片和文件夹的名字，文件夹的名字要和相机名字对应。