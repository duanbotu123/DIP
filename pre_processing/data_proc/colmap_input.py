import os
import cv2
import shutil
import argparse

def copy_and_rename_videos(src_dir, video_dir):
    os.makedirs(video_dir, exist_ok=True)
    video_files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))])

    for idx, file in enumerate(video_files):
        new_name = f"{idx+1:02d}.mp4"
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(video_dir, new_name)
        shutil.copy2(src_path, dst_path)
        print(f"Copied and renamed: {file} -> {dst_path}")

def extract_frames(video_dir, output_dir, frame_position=30):
    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')])

    for idx, video_file in enumerate(video_files):
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()

        if ret:
            out_name = os.path.join(output_dir, f"{idx+1:02d}.jpg")
            cv2.imwrite(out_name, frame)
            print(f"Saved frame: {out_name}")
        else:
            print(f"Failed to read frame from: {video_file}")
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy and rename videos, then extract one frame per video.")
    parser.add_argument("input_path", help="Path to folder containing original video files")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    print(input_path)
    videos_path = os.path.join(input_path, "videos")
    images_path = os.path.join(input_path, "colmap_images")

    copy_and_rename_videos(input_path, videos_path)
    print(f"Videos copied to: {videos_path}")
    extract_frames(videos_path, images_path)