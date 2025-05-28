export CUDA_VISIBLE_DEVICES=0
export motion_dir="/nas_data/dataset/4dhoi/25_04_03/person00/motion07"
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

cd "$motion_dir/videos" || exit

shopt -s nullglob

mapfile -t VIDEO_NAMES < <(
printf '%s\n' *.mp4 |      # 列出文件
sort -V |                  # 按数字自然排序
sed 's/\.mp4$//'           # 去掉后缀，只留 01 02 …
)

printf '(%s)\n' "${VIDEO_NAMES[*]}"

conda activate portrait_magic
cd /nas_data/home/hlp/code/4dhoi/HOI_Gen/human_tracking/SmplxTracking_241216

for item in "${VIDEO_NAMES[@]}"; do
    mkdir -p $motion_dir/images/$item
    save_folder=$motion_dir/images/$item
    python main_video_osot.py --video_path $motion_dir/videos/$item.mp4 --output_dir $save_folder --run_mode 0 --tracking_mode body --input_type video --debug
done

python main_video_osot.py --output_dir $motion_dir/images --run_mode 1 --tracking_mode body --input_type multi_view --sub_vis 01 09 14 24 28 31 --render
