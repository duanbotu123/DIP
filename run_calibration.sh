DATASET_PATH=/home/hlp/data/vton/zf/camera

# preprocess raw videos to fit colmap input
python /home/hlp/code/4dhoi/HOI_Gen/pre_processing/data_proc/colmap_input.py $DATASET_PATH

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/colmap_images

colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/colmap_images \
    --output_path $DATASET_PATH/sparse


__conda_setup="$('/nas_data/home/ycz/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nas_data/home/ycz/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/nas_data/home/ycz/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/nas_data/home/ycz/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate hoiG

# source ./human_tracking/mocap.sh /nas_data/home/ycz/hoi-test225/hoi-frame225


cd human_tracking/EasyMocap

python3 scripts/preprocess/extract_video.py $DATASET_PATH --no2d

python3 apps/calibration/detect_chessboard.py $DATASET_PATH --out $DATASET_PATH/output/calibration --pattern 7,6 --grid 0.134

python ./apps/calibration/read_colmap.py  $DATASET_PATH/sparse/0 .bin

python3 apps/calibration/align_colmap_ground.py $DATASET_PATH/sparse/0 $DATASET_PATH --plane_by_chessboard $DATASET_PATH --noshow

ROOT_DIR=/nas_data/dataset/4dhoi/25_04_03
CAMERA=/nas_data/dataset/4dhoi/ground_25_04_03

# find "$ROOT_DIR" -type d -name "avatar" | while read -r avatar_dir; do
#     cp "$CAMERA"/*.yml "$avatar_dir"
# done

# find "$ROOT_DIR" -type d -name "motion*" | while read -r motion_dir; do
#     cp "$CAMERA"/*.yml "$motion_dir"
# done