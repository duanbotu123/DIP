##  2D model

Download yolov4.weights and place it into data/models/yolov4.weights.

```
mkdir -p ./human_tracking/EasyMocap/data/models

wget -P ./human_tracking/EasyMocap/data/models https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights


```

Download pretrained HRNet weight and place it into data/models/pose_hrnet_w48_384x288.pth.

```
wget -P ./human_tracking/EasyMocap/data/models https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights


```

```
data
└── models
    ├── smpl_mean_params.npz
    ├── spin_checkpoint.pt
    ├── pose_hrnet_w48_384x288.pth
    └── yolov4.weights 
```