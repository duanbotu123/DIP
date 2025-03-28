export CUDA_VISIBLE_DEVICES=0
id_name=$1
save_folder=/nas_data/home/hlp/data/smpl_recon_test


## aios: initialize smplx
# REAL_save_folder=$(realpath ${save_folder})


# python -m torch.distributed.launch \
#     --nproc_per_node 1 \
#     --master_port 29200 \
#     --use_env \
#     main.py \
#     -c "config/aios_smplx_inference.py" \
#     --options batch_size=8 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
#     --resume "data/checkpoint/aios_checkpoint.pth" \
#     --eval \
#     --inference \
#     --to_vid \
#     --inference_input $REAL_save_folder/01.mp4 \
#     --output_dir $REAL_save_folder/aios

cd ../SmplxTracking_241216
cp -r $save_folder/aios/01_out/predictions $save_folder/smplx_recon

python main_video_osot.py --video_path $save_folder/01.mp4 --output_dir $save_folder --run_mode 0 --tracking_mode body --input_type video
python main_video_osot.py --output_dir $save_folder --run_mode 1 --tracking_mode body --input_type image
