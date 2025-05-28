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

export CUDA_VISIBLE_DEVICES=2
export VIDEO_DIR="/nas_data/dataset/4dhoi/25_04_03/person04"
export FRAME_RATE=60
export DEBUG=1

conda activate sam2

echo "Searching for 'motion*' directories in: $VIDEO_DIR"
# 1. 将所有找到的 'motion*' 目录路径保存到一个数组
mapfile -t motion_dirs_array < <(find "$VIDEO_DIR" -mindepth 1 -maxdepth 1 -type d -name "motion*")

# 检查是否找到了任何目录
if [ ${#motion_dirs_array[@]} -eq 0 ]; then
    echo "No directories matching 'motion*' found in '$VIDEO_DIR'."
    exit 0
fi

# 2. 打印数组中的所有路径
echo "Found the following directories to process:"
printf '%s\n' "${motion_dirs_array[@]}"
echo "--------------------------------------------------"

for motion_dir in "${motion_dirs_array[@]}"; do
    echo "Processing motion directory: $motion_dir"
    
    python /nas_data/home/hlp/code/4dhoi/HOI_Gen/pre_processing/sam2/semantic_seg.py --video "$motion_dir"

    if $DEBUG; then
        cd "$motion_dir/videos" || exit

        shopt -s nullglob

        mapfile -t VIDEO_NAMES < <(
        printf '%s\n' *.mp4 |      # 列出文件
        sort -V |                  # 按数字自然排序
        sed 's/\.mp4$//'           # 去掉后缀，只留 01 02 …
        )

        printf '(%s)\n' "${VIDEO_NAMES[*]}"
        for video_name in "${VIDEO_NAMES[@]}"; do
            echo "Processing video: $video_name"
            ffmpeg -framerate $FRAME_RATE -i $motion_dir/images/$video_name/masks/%06d.jpg \
            -c:v libx264 -pix_fmt yuv420p -r 30 \
            -y $motion_dir/images/$video_name/debug/masks.mp4
            echo "Video saved to $motion_dir/images/$video_name/debug/masks.mp4"
        done
        
        fi


