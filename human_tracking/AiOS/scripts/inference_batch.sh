# batch 处理 xj cvpr 2025

# xj
# INPUT_VIDEO=$1
# OUTPUT_DIR=$2

# python -m torch.distributed.launch \
#     --nproc_per_node 1 \
#     --master_port 29200 \
#     main_batch.py \
#     -c "config/aios_smplx_inference.py" \
#     --options batch_size=8 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
#     --resume "data/checkpoint/aios_checkpoint.pth" \
#     --eval \
#     --inference \
#     --inference_input /home/juyonggroup/xiangjun/git/PortraitMagic_new/data/chalamet/novel \
#     --output_dir /home/juyonggroup/xiangjun/git/PortraitMagic_new/data/chalamet/aios


python main_batch.py \
    -c "config/aios_smplx_inference.py" \
    --options batch_size=8 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
    --resume "data/checkpoint/aios_checkpoint.pth" \
    --eval \
    --inference \
    --inference_input /home/juyonggroup/xiangjun/git/PortraitMagic_new/data/chalamet/novel \
    --output_dir /home/juyonggroup/xiangjun/git/PortraitMagic_new/data/chalamet/aios