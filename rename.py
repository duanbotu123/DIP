import os
import argparse
import re

def rename_images(folder, dry_run=False):
    files = sorted(os.listdir(folder))
    pattern = re.compile(r'^(\d{8})(\.\w+)$')  # 匹配 8 位数字加扩展名

    for file in files:
        match = pattern.match(file)
        if match:
            old_path = os.path.join(folder, file)
            number_str, ext = match.groups()
            new_filename = f"{int(number_str):06d}{ext}"
            new_path = os.path.join(folder, new_filename)

            if dry_run:
                print(f"[Dry run] {file} → {new_filename}")
            else:
                os.rename(old_path, new_path)
                print(f"Renamed: {file} → {new_filename}")
        else:
            print(f"Skipped: {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename image files from %08d to %06d format.")
    parser.add_argument("folder", type=str, help="Path to the folder containing image files.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without renaming files.")
    args = parser.parse_args()

    rename_images(args.folder, args.dry_run)
