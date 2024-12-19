#!/bin/bash


# 遍历列表
for ((i=1; i<=14; i++))
do
    # 定义文件夹路径
    image_dir="./../../../test_data/human/images/$i"
    mask_dir="/data1/hlp/dataset/241129/human/mask/$i"
    annots_dir="/data1/hlp/dataset/241129/human/annots/$i"
    checkpoint_dir="./../checkpoints/sam2.1_hiera_large.pt"
    
    
    # 检查文件夹是否存在
    if [ -d "$image_dir" ] && [ -d "$annots_dir" ]; then
        # 执行 Python 脚本
        python /data1/hlp/workspace/Tracking/sam2/scripts/image_predict.py \
            --image "$image_dir" \
            --mask "$mask_dir" \
            --annots "$annots_dir" \
            --checkpoint $checkpoint_dir
        
        # 输出当前执行的任务信息
        echo "Processed folder $i"
    else
        # 如果文件夹不存在，跳过并输出提示信息
        echo "Skipping folder $i, one or more directories do not exist."
    fi
done

conda remove --name myenv --all