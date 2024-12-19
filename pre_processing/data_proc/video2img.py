import cv2
import os

# 视频文件夹路径
video_folder = './test_data/human/'
# 图片输出文件夹路径
output_folder = './test_data/human_image/'

# 获取视频文件列表
video_files = [f for f in os.listdir(video_folder) if f.endswith('.MP4')]

# 遍历每个视频文件
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    
    # 创建输出文件夹
    video_index = video_file.split('_')[0]  # 获取视频编号
    output_path = os.path.join(output_folder, video_index)
    os.makedirs(output_path, exist_ok=True)  # 创建文件夹，如果已存在则不报错

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()  # 读取一帧
        if not ret:  # 如果没有读取到帧，结束循环
            break
        
        # 保存帧为图片
        frame_filename = os.path.join(output_path, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()  # 释放视频对象

print("所有视频帧已成功提取并保存。")