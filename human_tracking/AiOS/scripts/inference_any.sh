
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
# INPUT_VIDEO=$1
# OUTPUT_DIR=$2
id_name=$1

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 29200 \
    main.py \
    -c "config/aios_smplx_inference.py" \
    --options batch_size=8 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
    --resume "data/checkpoint/aios_checkpoint.pth" \
    --eval \
    --inference \
    --to_vid \
    --inference_input /home/juyonggroup/xiangjun/git/BodyTracking_241216/data_video/YM2/YM2.mp4 \
    --output_dir /home/juyonggroup/xiangjun/git/BodyTracking_241216/data_video/YM2/aios

    # --inference_input /home/juyonggroup/xiangjun/git/BodyTracking_osot/data_sparse/${id_name}/novel_body/seq1/video.mp4 \
    # --output_dir /home/juyonggroup/xiangjun/git/BodyTracking_osot/data_sparse/${id_name}/novel_body/seq1/aios
