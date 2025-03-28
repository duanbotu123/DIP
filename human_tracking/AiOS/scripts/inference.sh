
# INPUT_VIDEO=$1
# OUTPUT_DIR=$2

# python -m torch.distributed.launch \
#     --nproc_per_node 2 \
#     main.py \
#     -c "config/aios_smplx_inference.py" \
#     --options batch_size=8 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
#     --resume "data/checkpoint/aios_checkpoint.pth" \
#     --eval \
#     --inference \
#     --to_vid \
#     --inference_input demo/${INPUT_VIDEO}.mp4 \
#     --output_dir demo/${OUTPUT_DIR}

# export OPENBLAS_NUM_THREADS=4
# export GOTO_NUM_THREADS=4
# export OMP_NUM_THREADS=4
# xj
INPUT_VIDEO=$1
OUTPUT_DIR=$2

# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch \
#     --nproc_per_node 1 \
#     --master_port 29200 \
#     main.py \
#     -c "config/aios_smplx_inference.py" \
#     --options batch_size=8 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
#     --resume "data/checkpoint/aios_checkpoint.pth" \
#     --eval \
#     --inference \
#     --to_vid \
#     --inference_input demo/${INPUT_VIDEO}.mp4 \
#     --output_dir demo/${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 29200 \
    main.py \
    -c "config/aios_smplx_inference.py" \
    --options batch_size=8 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
    --resume "data/checkpoint/aios_checkpoint.pth" \
    --eval \
    --inference \
    --to_vid \
    --inference_input demo/osot/${INPUT_VIDEO}.mp4 \
    --output_dir demo/osot/${OUTPUT_DIR}
