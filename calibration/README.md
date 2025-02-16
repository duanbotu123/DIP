参考[easymocap](https://github.com/zju3dv/EasyMocap/blob/master/apps/calibration/Readme.md)
# 1.Rename

相机输出格式和easymocap输入命名格式不同，使用rename.py重命名图片

```
python script.py image $data
# 修改$data文件夹中的子文件夹和子文件夹中的图片名
python script.py video $data
# 修改视频名

```

# 2.Detect the chessboard
```
# extract 2d
python3 scripts/preprocess/extract_video.py ${data} --no2d
# detect chessboard
python3 apps/calibration/detect_chessboard.py ${data} --out ${data}/output/calibration --pattern 9,6 --grid 0.1
```

The results will be saved in ```${data}/chessboard```, the visualization will be saved in ```${data}/output/calibration```.


# 3.Intrinsic Parameter Calibration
```
python3 apps/calibration/calib_intri.py ${data}
```

After the script finishes, you'll get ```intri.yml``` under ```${data}/output```.

# 4. Extrinsic Parameter Calibration
```
python3 apps/calibration/calib_extri.py ${extri} --intri ${intri}/output/intri.yml
```

After the script finishes, you'll get ```extri.yml``` under ```${intri}/output```.

# 5.Check the calibration
Check the results with a cube:
```
python3 apps/calibration/check_calib.py ${data} --out ${data}/output --mode cube --write
```
You'll get results in ```$data/output/cube```

# 6.Merge cameras
由于场地太大不能一次性标完，如果分别标定还需要运行将不同批次的相机外参变到同一个坐标系中。
```
python merge_calib.py --ex1 $extri1 --ex2 $extri2 --output $output
```
这是把第二个外参的坐标系变到第一个坐标系中，输出的也是yml格式。

# tips
1. 可以用环境变量extri,intri代替data

2. 如果遇到cv2.error: OpenCV(4.10.0) /io/opencv/modules/imgproc/src/color.cpp:196: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'

    改一下报错位置的代码，直接跳过这张图片，可能是图片损坏了

3. delete.sh用于删除未出现的棋盘格的图片，不然easymocap标定太慢了。