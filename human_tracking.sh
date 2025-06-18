export CUDA_VISIBLE_DEVICES=3
export VIDEO_DIR="/nas_data/dataset/4dhoi/25_04_03/person04"
export FRAME_RATE=60

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/apps/easybuild/software/Anaconda3/2024.02-1/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/apps/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh" ]; then
        . "/opt/apps/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh"
    else
        export PATH="/opt/apps/easybuild/software/Anaconda3/2024.02-1/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

port_offset_counter=0

# # 遍历所有名为 'avatar' 的目录
# find "$VIDEO_DIR" -type d -name "avatar" | while read -r avatar_dir; do
#     current_master_port=$((29200 + port_offset_counter))
#     echo "Processing motion directory: $avatar_dir with master_port: $current_master_port"

#     cd "$avatar_dir/videos" || exit

#     shopt -s nullglob

#     mapfile -t VIDEO_NAMES < <(
#     printf '%s\n' *.mp4 |      # 列出文件
#     sort -V |                  # 按数字自然排序
#     sed 's/\.mp4$//'           # 去掉后缀，只留 01 02 …
#     )

#     printf '(%s)\n' "${VIDEO_NAMES[*]}"
    
#     ## aios: initialize smplx
#     cd /nas_data/home/hlp/code/4dhoi/HOI_Gen/human_tracking/AiOS
#     conda activate aios

#     python -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port "$current_master_port" \
#         main.py \
#         -c "config/aios_smplx_inference.py" \
#         --options batch_size=4 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
#         --resume "data/checkpoint/aios_checkpoint.pth" \
#         --eval \
#         --inference \
#         --to_vid \
#         --inference_input $avatar_dir/videos/${VIDEO_NAMES[0]}.mp4 \
#         --output_dir $avatar_dir/aios

#     ## tracking
#     conda activate portrait_magic
#     cd /nas_data/home/hlp/code/4dhoi/HOI_Gen/human_tracking/SmplxTracking_241216
#     cp -r $avatar_dir/aios/${VIDEO_NAMES[0]}_out/predictions $avatar_dir/smplx_recon


#     for item in "${VIDEO_NAMES[@]}"; do
#         mkdir -p $avatar_dir/images/$item
#         save_folder=$avatar_dir/images/$item
#         python main_video_osot.py --video_path $avatar_dir/videos/$item.mp4 --output_dir $save_folder --run_mode 0 --tracking_mode body --input_type video
#     done

#     python main_video_osot.py --output_dir $avatar_dir/images --run_mode 1 --tracking_mode body --input_type multi_view --sub_vis 01 09 14 24 28 31 --render
#     port_offset_counter=$((port_offset_counter + 1))
# done

# echo "Searching for 'motion*' directories in: $VIDEO_DIR"
# # 1. 将所有找到的 'motion*' 目录路径保存到一个数组
# mapfile -t motion_dirs_array < <(find "$VIDEO_DIR" -mindepth 1 -maxdepth 1 -type d -name "motion*")

# # 检查是否找到了任何目录
# if [ ${#motion_dirs_array[@]} -eq 0 ]; then
#     echo "No directories matching 'motion*' found in '$VIDEO_DIR'."
#     exit 0
# fi

# # 2. 打印数组中的所有路径
# echo "Found the following directories to process:"
# printf '%s\n' "${motion_dirs_array[@]}"
# echo "--------------------------------------------------"

# # 3. 遍历数组中的每个路径进行处理
# for motion_dir in "${motion_dirs_array[@]}"; do
    motion_dir=/home/hlp/data/vton/zf
    current_master_port=$((29400 + port_offset_counter))
    echo "Processing motion directory: $motion_dir with master_port: $current_master_port"
    
    cd "$motion_dir/videos" || exit

    shopt -s nullglob

    mapfile -t VIDEO_NAMES < <(
    printf '%s\n' *.mp4 |      # 列出文件
    sort -V |                  # 按数字自然排序
    sed 's/\.mp4$//'           # 去掉后缀，只留 01 02 …
    )

    printf '(%s)\n' "${VIDEO_NAMES[*]}"

    ## aios: initialize smplx
    cd /nas_data/home/hlp/code/4dhoi/HOI_Gen/human_tracking/AiOS
    conda activate aios


    python -m torch.distributed.launch \
        --nproc_per_node 1 \
        --master_port "$current_master_port" \
        main.py \
        -c "config/aios_smplx_inference.py" \
        --options batch_size=8 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
        --resume "data/checkpoint/aios_checkpoint.pth" \
        --eval \
        --inference \
        --to_vid \
        --inference_input $motion_dir/videos/${VIDEO_NAMES[0]}.mp4 \
        --output_dir $motion_dir/aios

    ## tracking
    conda activate portrait_magic
    cd /nas_data/home/hlp/code/4dhoi/HOI_Gen/human_tracking/SmplxTracking_241216
    cp -r $motion_dir/aios/${VIDEO_NAMES[0]}_out/predictions $motion_dir/smplx_recon


    for item in "${VIDEO_NAMES[@]}"; do
        mkdir -p $motion_dir/images/$item
        save_folder=$motion_dir/images/$item
        python main_video_osot.py --video_path $motion_dir/videos/$item.mp4 --output_dir $save_folder --run_mode 0 --tracking_mode body --input_type video
    done

    python main_video_osot.py --output_dir $motion_dir/images --run_mode 1 --tracking_mode body --input_type multi_view --sub_vis 01 09 14 24 28 31 --render
    mkdir -p $motion_dir/thuman4/
    cp $motion_dir/body_track/smpl_params.npz $motion_dir/thuman4/
# done

