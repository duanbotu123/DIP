import os
import sys

def rename_folders_and_images(base_dir):
    subfolders = [f for f in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, f))]
    
    for idx, folder in enumerate(subfolders, start=1):
        new_folder_name = f"{idx:02d}"
        old_folder_path = os.path.join(base_dir, folder)
        new_folder_path = os.path.join(base_dir, new_folder_name)
        
        if old_folder_path != new_folder_path:
            os.rename(old_folder_path, new_folder_path)
        
        images = [f for f in sorted(os.listdir(new_folder_path)) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        
        for img_idx, img in enumerate(images, start=1):
            ext = os.path.splitext(img)[1].lower()
            new_img_name = f"{img_idx:06d}{ext}"
            old_img_path = os.path.join(new_folder_path, img)
            new_img_path = os.path.join(new_folder_path, new_img_name)
            
            if old_img_path != new_img_path:
                os.rename(old_img_path, new_img_path)

def rename_videos(base_dir):
    videos = [f for f in sorted(os.listdir(base_dir)) if f.lower().endswith(('mp4', 'avi', 'mov', 'mkv'))]
    
    for idx, video in enumerate(videos, start=1):
        ext = os.path.splitext(video)[1].lower()
        new_video_name = f"{idx:02d}{ext}"
        old_video_path = os.path.join(base_dir, video)
        new_video_path = os.path.join(base_dir, new_video_name)
        print(f'{old_video_path} -> {new_video_path}')
        
        if old_video_path != new_video_path:
            os.rename(old_video_path, new_video_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <image|video> <directory>")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    directory = sys.argv[2]
    
    if mode == "image":
        rename_folders_and_images(directory)
    elif mode == "video":
        rename_videos(directory)
    else:
        print("Invalid mode. Use 'image' or 'video'.")
        sys.exit(1)