#!/bin/bash

for i in {1..32}; do
    # 将数字格式化为两位数（例如：1→01，12→12）
    dir_num=$(printf "%02d" $i)
    
    # 构建目标路径
    target_dir="/nas_data/home/hlp/data/calibration2/extri/3/images/${dir_num}"
    
    # 执行Python命令
    python /nas_data/home/hlp/data/calibration2/delete.py "$target_dir" 1 300
done