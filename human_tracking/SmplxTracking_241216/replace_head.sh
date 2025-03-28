### bash replace_head.sh target_video.mp4 source_img.jpg result_body.mp4 result_body.pth out_folder device_id
export CUDA_VISIBLE_DEVICES=$6
export TORCH_HOME=/nas_ssd/yudong/torch_home
export HF_HOME=/nas_ssd/yudong/hf_home
target_video=$1
source_img=$2
result_body_video=$3
result_body_pth=$4
save_folder=$5
save_body_dir=$save_folder/body
save_head_dir=$save_folder/head


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/nas_ssd/yudong/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nas_ssd/yudong/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/nas_ssd/yudong/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/nas_ssd/yudong/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate portrait_magic

python main_video.py --video_path $result_body_video --output_dir $save_body_dir --run_mode 0
mkdir $save_body_dir/smplx_recon
cp $result_body_pth $save_body_dir/smplx_recon/smplx_recon.pth
python main_video.py --video_path $result_body_video --output_dir $save_body_dir --run_mode 1
conda activate 3DGS
python main_video.py --video_path $result_body_video --output_dir $save_body_dir --run_mode 2

mkdir $save_head_dir
mkdir $save_head_dir/source_img
cp $source_img $save_head_dir/source_img
mkdir $save_head_dir/smplx_recon
cp $save_body_dir/body_track/smplx_track.pth $save_head_dir/smplx_recon/

REAL_save_head_dir=$(realpath ${save_head_dir})
mkdir $save_head_dir/target_imgs
ffmpeg -y -i $target_video -q:v 0 -start_number 0 $save_head_dir/target_imgs/%6d.jpg
cd ../Portrait-4D/data_preprocess

conda activate 3DGS
python preprocess_dir.py --input_dir=$REAL_save_head_dir/source_img --save_dir=$REAL_save_head_dir/source_img_processed
python bfm2flame_simplified.py --input_dir=$REAL_save_head_dir/source_img_processed --save_dir=$REAL_save_head_dir/source_img_processed
python preprocess_dir.py --input_dir=$REAL_save_head_dir/target_imgs --save_dir=$REAL_save_head_dir/target_imgs_processed
python bfm2flame_simplified.py --input_dir=$REAL_save_head_dir/target_imgs_processed --save_dir=$REAL_save_head_dir/target_imgs_processed
cd ../portrait4d
python gen_images_portrait4d.py --network=./pretrained_models/portrait4d-v2-vfhq512.pkl \
	--srcdir=$REAL_save_head_dir/source_img_processed \
	--tardir=$REAL_save_head_dir/target_imgs_processed \
	--outdir=$REAL_save_head_dir/driving_head \
	--use_simplified=1

cd ../../BodyTracking

conda activate portrait_magic
ffmpeg -y -framerate 30 -i $save_head_dir/driving_head/645/%05d.jpg -filter:v "crop=512:512:512:0" -crf 10 $save_head_dir/driving_head.mp4
python main_video.py --video_path $save_head_dir/driving_head.mp4 --output_dir $save_head_dir --run_mode 0 --tracking_mode head
python main_video.py --video_path $save_head_dir/driving_head.mp4 --output_dir $save_head_dir --run_mode 1 --tracking_mode head 

conda activate 3DGS
python main_video.py --video_path $result_body_video --output_dir $save_body_dir --run_mode 3

# python main_video.py --video_path $result_body_video --output_dir $save_body_dir --run_mode 4 --driving_track_path data/curry2_YM1FR30/driving_body/body_track/smplx_track.pth
