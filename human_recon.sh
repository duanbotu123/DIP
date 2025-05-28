CUDA_VISIBLE_DEVICES=6
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

avatar_dir="/nas_data/dataset/4dhoi/25_04_03/person05/avatar"
# process image

echo "Processing avatar directory: $avatar_dir"
set -eo pipefail
shopt -s nullglob          # *.jpg 如果不存在不会原样传递
IFS=$'\n'                  # for 循环中的路径整体当作一项


# --- 1. 收集一级子目录并排序（忽略 camXX 自己） ---
subdirs=()
for d in "$avatar_dir/images"/*/; do
    base=$(basename "$d")
    [[ $base =~ ^cam[0-9]{2}$ ]] && continue   # 跳过已有 camXX
    [[ -d $d ]] && subdirs+=("$base")
done

# 按自然数排序（01 < 3 < 10）
IFS=$'\n' sorted=($(printf '%s\n' "${subdirs[@]}" | sort -V))
unset IFS

# --- 2. 逐个处理 ---
if ((${#sorted[@]} == 0)); then
    echo "❌ 在 $avatar_dir 内未找到任何子目录，退出。"
    exit 1
fi

printf "共检测到 %d 个源目录：%s\n" "${#sorted[@]}" "${sorted[*]}"

cam_idx=0
for src_base in "${sorted[@]}"; do
    src_dir="$avatar_dir/images/$src_base/ori_imgs"
    [[ -d $src_dir ]] || { echo "⚠️  跳过 $src_dir （目录不存在）"; continue; }

    dest_dir="$avatar_dir/images/cam$(printf '%02d' "$cam_idx")"
    mkdir -p "$dest_dir"
    echo "➡️  $src_dir → $dest_dir"

    # 用 mapfile 获取排序后的图片路径
    mapfile -d '' -t images < <(
        find "$src_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0 | sort -z -V
    )
    
    counter=0
    for img in "${images[@]}"; do
        printf -v new_name "%08d.jpg" "$counter"
        mv -- "$img" "$dest_dir/$new_name" || echo "❌ 复制失败: $img"
        ((counter++)) || true
    done

    printf "   ✅ 已移动 %d 张图片到 %s\n" "$counter" "$(basename "$dest_dir")"
    ((cam_idx++)) || true
done


echo "✅ 全部完成！共生成 $cam_idx 个 cam 目录。"



# process mask

conda activate birefnet
cd /home/hlp/code/4dhoi/HOI_Gen/pre_processing/BiRefNet
python ./birefnet.py --input_dir $avatar_dir/images --output_dir $avatar_dir/masks

# process camera

conda activate angs

python /home/hlp/code/4dhoi/HOI_Gen/pre_processing/angs_camera.py --intri $avatar_dir/intri.yml --extri $avatar_dir/extri.yml --out $avatar_dir/calibration.json

# process missing files

touch $avatar_dir/missing_img_files.txt

# process poses

cp $avatar_dir/body_track/smpl_params.npz $avatar_dir/smpl_params.npz

# process config
# should do it manually

# # angs
cd /home/hlp/code/4dhoi/HOI_Gen/human_recon/AnimatableGaussians

OPENCV_IO_ENABLE_OPENEXR=1 python -m gen_data.gen_pos_maps -c $avatar_dir/config.yaml

python main_avatar.py -c $avatar_dir/config.yaml --mode=train