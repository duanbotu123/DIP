export CUDA_VISIBLE_DEVICES=7

export VIDEO_DIR="/nas_data/home/hlp/data/smpl_multi_recon"
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

cd "$VIDEO_DIR" || exit
video_names=()
for dir in */; do
    dir_name=$(basename "$dir") 
    # 判断目录名是否全为数字（例如 01、002、123）
    if [[ $dir_name =~ ^[0-9]+$ ]]; then
        video_names+=( "$dir_name" )
    fi
done


echo "video names: ${video_names[@]}"

# use ffmpeg to generate videos
for item in "${video_names[@]}"; do
    IMAGE_PREFIX=$(printf "%03d" "$item")
    IMAGE_PATTERN="${IMAGE_PREFIX}_0001_%05d.jpg"
    # IMAGE_PATTERN="%06d.jpg"
    ffmpeg -framerate "$FRAME_RATE" -i "$VIDEO_DIR/$item/$IMAGE_PATTERN" -c:v libx264 -pix_fmt yuv420p "$VIDEO_DIR/${item}.mp4"
    echo "video generated: $VIDEO_DIR/${item}.mp4"
done

# ## aios: initialize smplx

# cd /nas_data/home/hlp/code/4dhoi/HOI_Gen/human_tracking/AiOS
# conda activate aios
# which python
# python -m torch.distributed.launch \
#     --nproc_per_node 1 \
#     --master_port 29200 \
#     main.py \
#     -c "config/aios_smplx_inference.py" \
#     --options batch_size=8 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
#     --resume "data/checkpoint/aios_checkpoint.pth" \
#     --eval \
#     --inference \
#     --to_vid \
#     --inference_input $VIDEO_DIR/${video_names[0]}.mp4 \
#     --output_dir $VIDEO_DIR/aios

## tracking
conda activate portrait_magic
cd /nas_data/home/hlp/code/4dhoi/HOI_Gen/human_tracking/SmplxTracking_241216
# cp -r $VIDEO_DIR/aios/${video_names[0]}_out/predictions $VIDEO_DIR/smplx_recon
for item in "${video_names[@]}"; do
    mkdir -p $VIDEO_DIR/images/$item
    save_folder=$VIDEO_DIR/images/$item
    python main_video_osot.py --video_path $VIDEO_DIR/$item.mp4 --output_dir $save_folder --run_mode 0 --tracking_mode body --input_type video
done


# # python main_video_osot.py --output_dir $save_folder --run_mode 1 --tracking_mode body --input_type image
python main_video_osot.py --output_dir $VIDEO_DIR/images --run_mode 1 --tracking_mode body --input_type multi_view --sub_vis 01 09 14 24 28 31

## remove origin images
for item in "${video_names[@]}"; do
    rm -r $VIDEO_DIR/$item
done