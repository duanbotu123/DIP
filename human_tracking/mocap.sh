data=./test_data/human_image/001/

python3 ./human_tracking/EasyMocap/apps/preprocess/extract_keypoints.py ${data} --mode mp-holistic

python3 ./human_tracking/EasyMocap/apps/preprocess/extract_keypoints.py ${data} --mode yolo-hrnet --annot ${data}/annots_yolo