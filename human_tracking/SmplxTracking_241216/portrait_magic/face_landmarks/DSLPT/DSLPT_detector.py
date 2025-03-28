import cv2
from tqdm import tqdm
from .model import Dynamic_sparse_alignment_network
from .Config import cfg
import os
import torch
from .Dataloader import Folder_Dataset, Image_Dataset
from .utils import transform_pixel_v2
import ffmpeg
import numpy as np
from ...utils_funcs.video_writer import VideoWriter
from ...dmm_models import smplx_model_path

def check_ldmks_intersect(ldmks_a, ldmks_b):
    """
    check if two landmarks sets intersect
    """
    a_l, a_r, a_t, a_b = np.min(ldmks_a[:, 0]), np.max(ldmks_a[:, 0]), np.min(ldmks_a[:, 1]), np.max(ldmks_a[:, 1])
    b_l, b_r, b_t, b_b = np.min(ldmks_b[:, 0]), np.max(ldmks_b[:, 0]), np.min(ldmks_b[:, 1]), np.max(ldmks_b[:, 1])
    if (a_l*.5 + a_r*.5)<b_l or (a_l*.5 + a_r*.5)>b_r:
        return False
    if (a_t*.5 + a_b*.5)<b_t or (a_t*.5 + a_b*.5)>b_b:
        return False
    if (b_l*.5 + b_r*.5)<a_l or (b_l*.5 + b_r*.5)>a_r:
        return False
    if (b_t*.5 + b_b*.5)<a_t or (b_t*.5 + b_b*.5)>a_b:
        return False
    return True

class DSLPT_utils():
    def __init__(self):
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        dslpt_model = Dynamic_sparse_alignment_network(
            cfg.WFLW.NUM_POINT, cfg.MODEL.OUT_DIM, cfg.MODEL.TRAINABLE,
            cfg.MODEL.INTER_LAYER, cfg.TRANSFORMER.NHEAD, cfg.TRANSFORMER.FEED_DIM,
            os.path.join(file_dir_path, cfg.WFLW.INITIAL_PATH), cfg)
        
        checkpoint_file = os.path.join(file_dir_path, 'Weight/DSLPT_WFLW_12_layers.pth')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        dslpt_model.load_state_dict(checkpoint)
        self.dslpt_model = dslpt_model.cuda().eval()
        
        self.wflw_to_pts51 = np.loadtxt(os.path.join(smplx_model_path, 'wflw_to_pts51.txt'), dtype=np.int64)


    def run_folder_refine(self, imgs_folder, save_folder, debug_dir = 'none'):
        folder_dataset = Folder_Dataset(cfg, imgs_folder)
        folder_loader = torch.utils.data.DataLoader(
            folder_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=False
        )

        video_out = None

        for input, trans, img, img_names in tqdm(folder_loader):
            with torch.inference_mode():
                outputs = self.dslpt_model(input.cuda())[0]
            ldmks = outputs[2][:, -1].cpu().numpy() * 256.0
            trans = trans.numpy()
            show_ldmks = []
            show_colors = []
            for i in range(trans.shape[0]):
                ldmk_path = os.path.join(imgs_folder.replace('ori_imgs', 'ldmks'), img_names[i].replace('.jpg', '.wb'))
                if not os.path.isfile(ldmk_path):
                    show_ldmks.append(ldmks[i])
                    show_colors.append((0,0,255))
                    continue
                ldmks[i] = transform_pixel_v2(ldmks[i], trans[i], inverse=True)
                wb_ldmk = np.loadtxt(ldmk_path, dtype=np.float32)

                if not check_ldmks_intersect(wb_ldmk[23:91, :2], ldmks[i]):
                    # print('no intersection')
                    show_ldmks.append(ldmks[i])
                    show_colors.append((0,0,255))
                    continue
                
                wb_ldmk[40:91, :2] = ldmks[i][self.wflw_to_pts51]
                np.savetxt(ldmk_path, wb_ldmk, fmt='%f')
                show_ldmks.append(wb_ldmk[23:91, :2])
                show_colors.append((255,0,0))

            if debug_dir != 'none':
                img = img.numpy()
                for i in range(img.shape[0]):
                    for j in range(show_ldmks[i].shape[0]):
                        img[i] = cv2.circle(img[i], (int(show_ldmks[i][j, 0]), int(show_ldmks[i][j, 1])), 1, show_colors[i], 1)
                if video_out is None:
                    video_out = VideoWriter(os.path.join(debug_dir, 'face_ldmks_refine.mp4')) 
                for i in range(img.shape[0]):
                    video_out.write_frame(img[i, :, :, ::-1])
        
        if video_out is not None:
            video_out.close()

    def run_image(self, oriImg):
        ### oriImg: numpy.array RGB image -> (133, 3) xs,ys,scores
        self.folder_dataset_tmp = Image_Dataset(cfg)
        input, trans = self.folder_dataset_tmp.process_img(oriImg)
        with torch.inference_mode():
            outputs = self.dslpt_model(input.cuda().unsqueeze(0))[0]
        ldmks = outputs[2][:, -1].cpu().numpy() * 256.0
        trans = trans.numpy()
        ldmks[0] = transform_pixel_v2(ldmks[0], trans, inverse=True)
        return ldmks[0]

    def run_folder(self, imgs_folder, save_folder, debug_dir = 'none'):
        folder_dataset = Folder_Dataset(cfg, imgs_folder)
        folder_loader = torch.utils.data.DataLoader(
            folder_dataset,
            batch_size=10,
            shuffle=False,
            num_workers=10,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=False
        )

        video_out = None

        for input, trans, img, img_names in tqdm(folder_loader):
            with torch.inference_mode():
                outputs = self.dslpt_model(input.cuda())[0]
            ldmks = outputs[2][:, -1].cpu().numpy() * 256.0
            trans = trans.numpy()
            for i in range(trans.shape[0]):
                ldmks[i] = transform_pixel_v2(ldmks[i], trans[i], inverse=True)
                img_name = img_names[i]
                if img_name != 'none':
                    np.savetxt(os.path.join(save_folder, img_name.replace('.jpg', '.wflw')), ldmks[i], '%f')
            if debug_dir != 'none':
                img = img.numpy()
                for i in range(img.shape[0]):
                    for j in range(ldmks[i].shape[0]):
                        img[i] = cv2.circle(img[i], (int(ldmks[i][j, 0]), int(ldmks[i][j, 1])), 1, (0, 0, 255), 1)
                if video_out is None:
                    video_out = (ffmpeg.input('pipe:0', format='rawvideo', pix_fmt='rgb24', s=str(img.shape[2]) + 'x' + str(img.shape[1])).output(os.path.join(debug_dir, 'face_ldmks.mp4'), pix_fmt='yuv420p', vcodec='libx264', r=25).overwrite_output().run_async(pipe_stdin=True))
                for i in range(img.shape[0]):
                    video_out.stdin.write(np.array(img[i, :, :, ::-1]).tobytes())
        
        if video_out is not None:
            video_out.stdin.close()
            video_out.wait()
            
            
