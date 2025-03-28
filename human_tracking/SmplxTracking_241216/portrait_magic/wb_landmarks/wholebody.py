import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from .folder_loader import Folder_Dataset
from .onnxdet import inference_detector
from .onnxpose import inference_pose
from ..utils_funcs.video_writer import VideoWriter


class DWpose():
    def __init__(self):
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        onnx_pose = os.path.join(file_dir_path, 'ckpts/dw-ll_ucoco_384.onnx')
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=['CUDAExecutionProvider'])
        
        onnx_det = os.path.join(file_dir_path, 'ckpts/yolox_l.onnx')
        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=['CUDAExecutionProvider'])

    def run_folder(self, imgs_folder, save_folder, debug_dir='none'):
        folder_dataset = Folder_Dataset(imgs_folder)
        folder_loader = torch.utils.data.DataLoader(
            folder_dataset, batch_size=8, shuffle=False, num_workers=5, pin_memory=True, drop_last=False)
        video_out = None
        for img_rgb, img_names in tqdm(folder_loader, desc='whole-body landmarks'):
            for i in range(img_rgb.shape[0]):
                oriImg = img_rgb[i].numpy()
                det_result, is_detected = inference_detector(self.session_det, oriImg)
                keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)
                # print(keypoints.shape, scores.shape) (1, 133, 2), (1, 133)
                save_arr = np.concatenate((keypoints[0], scores[0].reshape(-1, 1)), axis=1)
                if is_detected:
                    np.savetxt(os.path.join(save_folder, img_names[i][:-4] + '.wb'), save_arr, fmt='%f')
                    np.savetxt(os.path.join(save_folder, img_names[i][:-4] + '.rect'), det_result[0], fmt='%f')
                
                if debug_dir != 'none':
                    ldmks = keypoints[0]
                    score = scores[0]
                    for j in range(ldmks.shape[0]):
                        if score[j] > .7:
                            cv2.circle(oriImg, (int(ldmks[j, 0]), int(ldmks[j, 1])), 1, (255, 0, 0), 1, lineType=cv2.FILLED)
                    cv2.rectangle(oriImg, (int(det_result[0, 0]), int(det_result[0, 1])), (int(det_result[0, 2]), int(det_result[0, 3])), (0, 0, 255), 3)
                    if video_out is None:
                        video_out = VideoWriter(os.path.join(debug_dir, 'ldmks.mp4'))
                    video_out.write_frame(oriImg)
        if video_out is not None:
            video_out.close()


    def run_image(self, oriImg):
        ### oriImg: numpy.array RGB image -> (133, 3) xs,ys,scores
        det_result, _ = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)
        return np.concatenate((keypoints[0], scores[0].reshape(-1, 1)), axis=1)