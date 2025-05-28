import os

def create_structure(parent_dir, num_persons, motion_list):
    # 创建父目录
    os.makedirs(parent_dir, exist_ok=True)
    print(f"Created parent directory: {parent_dir}")
    
    # 循环创建每个 person 子目录
    for i in range(num_persons):
        # 格式化 person 目录名称，例如 person00, person01, ...
        person_dir = os.path.join(parent_dir, f"person{str(i).zfill(2)}")
        os.makedirs(person_dir, exist_ok=True)
        print(f"Created person directory: {person_dir}")
        
        # 在 person 目录下创建 avatar 目录（不需要新建txt文件）
        avatar_dir = os.path.join(person_dir, "avatar")
        os.makedirs(avatar_dir, exist_ok=True)
        print(f"Created avatar directory: {avatar_dir}")
        
        # 在 person 目录下为每个 motion 目录创建文件夹，并在其中新建 txt 文件
        for motion in motion_list:
            motion_dir = os.path.join(person_dir, motion)
            os.makedirs(motion_dir, exist_ok=True)
            print(f"Created motion directory: {motion_dir}")
            
            # 指定在 motion 目录内新建 txt 文件的路径
            txt_file_path = os.path.join(motion_dir, "motion_text.txt")
            try:
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    # 写入一些默认内容，可根据需要修改
                    f.write(f"这是在 {motion_dir} 目录下创建的txt文件。\n")
                print(f"Created text file: {txt_file_path}")
            except Exception as e:
                print(f"Failed to create text file in {motion_dir}: {e}")

if __name__ == "__main__":
    # 父目录名称
    parent_directory = "25_03_31"
    # 假设需要创建 10 个 person 目录：person00 ~ person09
    number_of_persons = 6
    # 定义 motion 目录列表（可根据需要扩展）
    motion_dirs = []
    for i in range(1,21):
        motion_dirs.append(f'motion{i:02d}')

    print(motion_dirs)
    create_structure(parent_directory, number_of_persons, motion_dirs)
