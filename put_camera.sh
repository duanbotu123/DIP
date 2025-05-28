ROOT_DIR=/nas_data/dataset/4dhoi/25_04_03
CAMERA=/nas_data/dataset/4dhoi/ground_25_04_03

find "$ROOT_DIR" -type d -name "avatar" | while read -r avatar_dir; do
    cp "$CAMERA"/*.yml "$avatar_dir"
done

find "$ROOT_DIR" -type d -name "motion*" | while read -r motion_dir; do
    cp "$CAMERA"/*.yml "$motion_dir"
done