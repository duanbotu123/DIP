import os

def rename_subfolders(parent_folder):
    # 获取子文件夹列表，并按原始顺序排序
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    subfolders.sort()  # 按字母顺序排序

    for idx, folder in enumerate(subfolders):
        # 新的文件夹名称
        new_name = f'{(idx+1):02d}'
        old_path = os.path.join(parent_folder, folder)
        new_path = os.path.join(parent_folder, new_name)

        # 如果新名称已存在，跳过
        if os.path.exists(new_path):
            print(f"Skipping {folder}: {new_name} already exists.")
            continue

        # 重命名文件夹
        os.rename(old_path, new_path)
        print(f"Renamed {folder} to {new_name}")

# 使用示例
parent_folder = "/nas_data/home/hlp/data/hoi-test225/images"  # 替换为你的父文件夹路径
rename_subfolders(parent_folder)