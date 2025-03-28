import cv2
import numpy as np
import subprocess
import os

class VideoWriter():
    def __init__(self, filename, fps=25):
        self.filename = filename
        self.fps = fps
        # 基于提供的 filename 生成临时文件名
        self.temp_filename = f"{filename}_temp.mp4"
        self.writer = None

    def write_frame(self, frame):
        if self.writer is None:
            height, width, channels = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.temp_filename, fourcc, self.fps, (width, height))
        # 确保帧的颜色格式为 BGR（OpenCV 的默认格式）
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)
                
    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            # 使用 ffmpeg 将临时 AVI 文件重新编码为高质量的 MP4 文件
            command = [
                'ffmpeg',
                '-y',
                '-i', self.temp_filename,
                '-vcodec', 'libx264',
                '-crf', '15',
                '-pix_fmt', 'yuv420p',
                self.filename
            ]
            subprocess.run(command, check=True)
            # 删除临时文件
            os.remove(self.temp_filename)
