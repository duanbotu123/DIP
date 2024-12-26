#!/bin/bash

# 设置源目录和目标目录
SOURCE_DIR="./test_data/human_image/001"
TARGET_DIR="./test_data/human_image/001/images"

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

# 移动图片文件到目标目录
find "$SOURCE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -exec mv {} "$TARGET_DIR" \;

echo "所有图片已移动到 $TARGET_DIR 目录下。"
