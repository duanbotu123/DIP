# EasyMocap

EasyMocap用于相机标定和估计smpl参数

相机标定，输入内参标定多视角视频与外参标定多视角视频/图片
https://github.com/zju3dv/EasyMocap/blob/master/apps/calibration/Readme.md

估计smpl参数，输入相机参数、人体多视角视频
https://github.com/zju3dv/EasyMocap/blob/master/doc/quickstart.md

更详细的飞书记录文档
https://ucny77rqcmoi.feishu.cn/wiki/WmcBwUyn4idlGQkf1nBcW2DInUd

## 环境配置
官方文档
- https://chingswy.github.io/easymocap-public-doc/install/install.html
- https://github.com/zju3dv/EasyMocap/blob/master/doc/quickstart.md
主要按照第一个文档install的小节来然后一直装到visualization部分，部分SMPL models data在第二个文档

可能遇到的问题

- 准备SMPL models，这一部分有的地方没写清楚，可以直接下载链接：https://rec.ustc.edu.cn/share/95febaf0-c332-11ef-b0c5-7f780ff43eb9
将data放到./data
- numpy2.0.2,pytorch1.12.1,运行python3 apps/preprocess/extract_keypoints.py ${data} --mode yolo-hrnet会出错，改装numpy1.26.0可以解决问题

## 相机标定
官方文档
https://github.com/zju3dv/EasyMocap/blob/master/apps/calibration/Readme.md

- 数据组织
  内参所有视角单独视频，外参所有视角要看到一个静态棋盘格
    ```python
    <intri_data>
    └── videos
       ├── 1.mp4
       ├── 2.mp4
       ├── ...
        └── xx.mp4
    ```
    ```python
    <extri_data>
    └── videos
       ├── 1.mp4
       ├── 2.mp4
       ├── ...
        └── xx.mp4
    ```
- 分别对<intri_data>和<extri_data>检测棋盘
  - 视频切片成图片
    ```python
    python3 scripts/preprocess/extract_video.py ${data} --no2d
    ```
  - 检测棋盘
    ```python
    python3 apps/calibration/detect_chessboard.py ${data} --out ${data}/output/calibration --pattern 7,6 --grid 0.138
    ``` 
    pattern代表交点数，grid代表棋盘物理格子边长
    可以手动标棋盘但目前尝试结果误差较大

- 估计相机参数
  - 标内参
    ```python
    python3 apps/calibration/calib_intri.py ${data}
    ```
    结果出现在intri文件夹下
  - 标外参
    ```python
    python3 apps/calibration/calib_extri.py ${extri} --intri ${intri}/output/intri.yml
    ```
    结果出现在extri文件夹下


## 重建smplx
官方文档
https://github.com/zju3dv/EasyMocap/blob/master/doc/quickstart.md

- 组织数据,视频格式要.mp4小写不要.MP4,数字要和标定相机参数的时候对上
    ```python
    <seq>
    ├── intri.yml
    ├── extri.yml
    └── videos
        ├── 1.mp4
        ├── 2.mp4
        ├── ...
        ├── 8.mp4
        └── 9.mp4
    ```
    视频切片
    ```python
    python3 scripts/preprocess/extract_video.py ${data} --handface 
    ``` 
- 检测关键点(有多种方法)
    - Mediapipe估全身（身体不准）
    ```python
    python3 apps/preprocess/extract_keypoints.py ${data} --mode mp-holistic --annot annots1
    ``` 
    - YOLOv4+HRNet估身体
    ```python
    python3 apps/preprocess/extract_keypoints.py ${data} --mode yolo-hrnet --annot annots2
    ```
    - 合并结果
    ```python
    python3 ./scripts/my/merge_annotations.py ${data}
    ```
- smplx重建
  ```python
  python3 apps/demo/mv1p.py ${data} --out ${data}/output/smplx --vis_det --vis_repro --undis --sub_vis 1 4 7 13 --body bodyhandface --model smplx --gender male --vis_smpl
  ```

## 处理结果用于angs

- smpl对齐平移
  
  使用代码 https://ucny77rqcmoi.feishu.cn/wiki/ZBZ2wrtrti3NwXkqsEMcKBbrnHh
- 将smpl从json格式转npz格式
  
  使用代码 https://ucny77rqcmoi.feishu.cn/wiki/ZBZ2wrtrti3NwXkqsEMcKBbrnHh
- 相机参数转换
  
  使用代码 https://ucny77rqcmoi.feishu.cn/wiki/BnsqwKlCoiD1lGkQbGrcGc7qnVh