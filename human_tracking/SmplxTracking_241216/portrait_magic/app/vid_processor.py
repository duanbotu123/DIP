import os
import cv2
import torch
from ..wb_landmarks import DWpose
from ..face_filter import FaceFilter
from ..video_matting import RVMatter
from ..face_parsing import FaceParser
from ..body_tracking import LDMK_Fitting
from ..feature_tracking_all import CotrackOnlineFeature
from ..face_landmarks import DSLPT_utils
from ..iris_landmarks import MPIris
from ..normal_estimation import NormalEstimator

class VidProcessor():
    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.ldmk_utils = DWpose()
        self.filter_utils = FaceFilter()
        self.face_ldmk_utils = DSLPT_utils()
        self.iris_ldmk_utils = MPIris()
        self.face_parser = FaceParser()
        self.seg_utils = RVMatter()
        self.feature_extractor = CotrackOnlineFeature()
        self.ldmk_fiter = LDMK_Fitting()
        self.normal_estimator = NormalEstimator()

    def process_vid(self, video_path: str, save_dir: str, with_debug = False):
        debug_none_path = 'none'
        debug_path = os.path.join(save_dir, "debug")
        os.makedirs(debug_path, exist_ok=True)
        ori_imgs_dir = os.path.join(save_dir, "ori_imgs")

        # ### step 1: using ffmpeg to extract frames from video_path to ori_imgs_dir
        # cap_vid = cv2.VideoCapture(video_path)
        # video_width = int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        # video_height = int(cap_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # os.makedirs(ori_imgs_dir, exist_ok=True)
        # if (video_width%2==1) or (video_height%2==1):
        #     ffmpeg_cmd = 'ffmpeg -loglevel error -y -i ' + video_path + ' -s ' + str(video_width//2*2) + 'x' + str(video_height//2*2) + ' -q:v 0 -start_number 0 '+ ori_imgs_dir +'/%06d.jpg'
        # else:
        #     ffmpeg_cmd = 'ffmpeg -loglevel error -y -i ' + video_path + ' -q:v 0 -start_number 0 '+ ori_imgs_dir +'/%06d.jpg'
        # os.system(ffmpeg_cmd)

        # ### step 2: detect whole body landmarks with https://github.com/IDEA-Research/DWPose
        # ldmks_dir = os.path.join(save_dir, "ldmks")
        # os.makedirs(ldmks_dir, exist_ok=True)
        # self.ldmk_utils.run_folder(ori_imgs_dir, ldmks_dir, debug_path)

        # ### filter face to exclude bad videos, e.g.: not continous frames, too small faces, no/partial faces, etc...
        # is_good_video = self.filter_utils.filter_face(ori_imgs_dir, debug_path, with_crop = True)
        # if is_good_video < 0:
        #     print('bad video ' + ori_imgs_dir)
        #     return 
    
        # #### refine and replace face ldmks
        # face_ldmks_dir = os.path.join(save_dir, "face_ldmks")
        # os.makedirs(face_ldmks_dir, exist_ok=True)
        # self.face_ldmk_utils.run_folder_refine(ori_imgs_dir, face_ldmks_dir, debug_path)

        # #### iris landmarks
        # iris_ldmks_dir = os.path.join(save_dir, "iris_ldmks")
        # os.makedirs(iris_ldmks_dir, exist_ok=True)
        # self.iris_ldmk_utils.run_folder(ori_imgs_dir, iris_ldmks_dir, debug_path)

        # ### face parsing
        # parsing_dir = os.path.join(save_dir, "parsing")
        # os.makedirs(parsing_dir, exist_ok=True)
        # self.face_parser.run_folder(ori_imgs_dir, parsing_dir, debug_path)

        # ### step 3: segment portrait with RobustVideoMatting
        # seg_imgs_dir = os.path.join(save_dir, "seg_masks")
        # os.makedirs(seg_imgs_dir, exist_ok=True)
        # self.seg_utils.run_folder(ori_imgs_dir, seg_imgs_dir, debug_path)

        # ### normal map
        # normal_imgs_dir = os.path.join(save_dir, "normal")
        # os.makedirs(normal_imgs_dir, exist_ok=True)
        # self.normal_estimator.run_folder(ori_imgs_dir, normal_imgs_dir, debug_path)

        # ### tracking feature extraction
        # tracking_feature_dir = os.path.join(save_dir, "track_feature")
        # os.makedirs(tracking_feature_dir, exist_ok=True)
        # self.feature_extractor.run_folder(ori_imgs_dir, tracking_feature_dir, debug_path)

        ### step 5: fit smplx+d with differentiable rendering
        body_track_dir = os.path.join(save_dir, "body_track")
        os.makedirs(body_track_dir, exist_ok=True)
        self.ldmk_fiter.run_folder(ori_imgs_dir, body_track_dir, debug_path)
