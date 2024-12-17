import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

video_dir = "/data1/hlp/dataset/241124/videos/3.MP4"


with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(state)

    ann_frame_idx = 0
    ann_obj_id = 1
    box = np.array([1265.4900952380956,
                463.07230476190466,
                1894.3860952380953,
                2102.303504761904])
    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        inference_state=state, 
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
        )

    video_segments = {}
    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        video_segments[frame_idx] = {
        obj_id: (masks[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }