import os
import torch
from ..wb_landmarks import DWpose
from ..body_tracking import LDMK_Fitting,LDMK_Fitting_multiview,LDMK_Fitting_sparse
from ..save_for_training import TrainImgsWriter
from ..sapiens_pose import PoseEstimator

# import xxx

def process_portrait_video_osot(video_path: str, save_dir: str, with_debug = False, run_mode = 0, tracking_mode='body', input_type='video', sub_vis = None):
    """
    Process a video to extract recon infos and save them to save_dir
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if with_debug and run_mode == 0:
        debug_path = os.path.join(save_dir, "debug")
        os.makedirs(debug_path, exist_ok=True)
    elif with_debug and run_mode == 1 and input_type == 'multi_view':
        debug_path = save_dir.replace('images','debug')
        os.makedirs(debug_path, exist_ok=True)
    elif with_debug and run_mode == 1 and input_type != 'multi_view':
        debug_path = os.path.join(save_dir, "debug")
        os.makedirs(debug_path, exist_ok=True)
    else:
        debug_path = 'none'
    debug_none_path = 'none'

    ### step 1: using ffmpeg to extract frames from video_path to ori_imgs_dir
    ori_imgs_dir = os.path.join(save_dir, "ori_imgs")
    if run_mode == 0 and input_type == 'video':
        os.makedirs(ori_imgs_dir, exist_ok=True)
        ffmpeg_cmd = 'ffmpeg -y -i ' + video_path + ' -start_number 0 ' f' -q:v 0 ' + ori_imgs_dir + '/%06d.jpg'
        os.system(ffmpeg_cmd)

    if tracking_mode == 'body':
        from ..body_tracking import LDMK_Fitting
    else:
        from ..body_tracking_head import LDMK_Fitting

    if run_mode == 0:
        ## step 2: detect whole body landmarks with https://github.com/IDEA-Research/DWPose
        ldmks_dir = os.path.join(save_dir, "ldmks")
        ldmk_utils = DWpose()
        os.makedirs(ldmks_dir, exist_ok=True)
        ldmk_utils.run_folder(ori_imgs_dir, ldmks_dir, debug_path)

        # if tracking_mode == 'body':
        #     pose_estimator = PoseEstimator()
        #     pose_estimator.run_folder(ori_imgs_dir, ldmks_dir, debug_path)

        # #### refine and replace face ldmks
        # face_ldmk_utils = DSLPT_utils()
        # face_ldmks_dir = os.path.join(save_dir, "face_ldmks")
        # os.makedirs(face_ldmks_dir, exist_ok=True)
        # face_ldmk_utils.run_folder_refine(ori_imgs_dir, face_ldmks_dir, debug_path)

        # #### iris landmarks
        # iris_ldmk_utils = MPIris()
        # iris_ldmks_dir = os.path.join(save_dir, "iris_ldmks")
        # os.makedirs(iris_ldmks_dir, exist_ok=True)
        # iris_ldmk_utils.run_folder(ori_imgs_dir, iris_ldmks_dir, debug_path)

        # ### face parsing
        # face_parser = FaceParser()
        # parsing_dir = os.path.join(save_dir, "parsing")
        # os.makedirs(parsing_dir, exist_ok=True)
        # face_parser.run_folder(ori_imgs_dir, parsing_dir, debug_path)

        # ### step 3: segment portrait with RVM
        # if tracking_mode == 'body':
        #     seg_utils = BiRefNet()
        # else:
        #     seg_utils = RVMatter()
        # seg_imgs_dir = os.path.join(save_dir, "seg_masks")
        # os.makedirs(seg_imgs_dir, exist_ok=True)
        # seg_utils.run_folder(ori_imgs_dir, seg_imgs_dir, debug_path)

        # # normal_estimator = NormalEstimator()
        # # normal_imgs_dir = os.path.join(save_dir, "normal")
        # # os.makedirs(normal_imgs_dir, exist_ok=True)
        # # normal_estimator.run_folder(ori_imgs_dir, normal_imgs_dir, debug_path)
        
        # ### tracking feature extraction
        # feature_extractor = CotrackOnlineFeature()
        # tracking_feature_dir = os.path.join(save_dir, "track_feature")
        # os.makedirs(tracking_feature_dir, exist_ok=True)
        # feature_extractor.run_folder(ori_imgs_dir, tracking_feature_dir, debug_path)

    elif run_mode == 1:
        if input_type == 'multi_view':
            ldmk_fiter = LDMK_Fitting_multiview()
            body_track_dir = save_dir.replace('images','body_track')
            os.makedirs(body_track_dir, exist_ok=True)
            ldmk_fiter.run_folder(save_dir, body_track_dir, debug_path, sub_vis)
        elif input_type != 'multi_view':
            ldmk_fiter = LDMK_Fitting()
            body_track_dir = os.path.join(save_dir, "body_track")
            os.makedirs(body_track_dir, exist_ok=True)
            ldmk_fiter.run_folder(ori_imgs_dir, body_track_dir, debug_path)
        else:
            ldmk_fiter = LDMK_Fitting_sparse()

        if tracking_mode == 'head':
            training_imgs_dir = os.path.join(save_dir, "train_imgs")
            os.makedirs(training_imgs_dir, exist_ok=True)
            train_imgs_writer = TrainImgsWriter()
            train_imgs_writer.run_folder(ori_imgs_dir, training_imgs_dir, debug_path) 

