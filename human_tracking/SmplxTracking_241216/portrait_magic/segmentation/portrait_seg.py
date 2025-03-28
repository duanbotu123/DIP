from .U2Net.model import U2NET
import torch
import os
from .U2Net.data_loader import RescaleT, ToTensorLab, SalObjDataset
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from skimage import io
import numpy as np
import cv2
import ffmpeg

# from segment_anything import SamPredictor, sam_model_registry
# from pytorch3d.ops import sample_farthest_points

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(os.path.join(d_dir, imidx+'.png'))

def largest_connected(image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2 > 155



class PortraitSeg():
    def __init__(self):
        self.model = U2NET(3, 1)
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(
            file_dir_path, 'U2Net/saved_models/u2net_human_seg.pth')
        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda().eval()

        ### comment sam model since no benefits
        # sam_model = sam_model_registry["vit_l"](checkpoint=os.path.join(os.path.expanduser('~'), ".sam_models/sam_hq_vit_l.pth")).cuda()
        # self.sam_predictor = SamPredictor(sam_model)
        # self.blur_op = transforms.GaussianBlur(kernel_size=13, sigma=0.3)

    def run_folder(self, imgs_folder, save_folder, debug_dir='none'):
        img_paths = natsorted([os.path.join(imgs_folder, f)
                               for f in os.listdir(imgs_folder) if f.endswith('.jpg')])
        test_salobj_dataset = SalObjDataset(img_name_list=img_paths, lbl_name_list=[],
                                            transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
        test_salobj_dataloader = DataLoader(
            test_salobj_dataset, batch_size=10, shuffle=False, num_workers=10, drop_last=False)
        video_out = None
        for data_test in tqdm(test_salobj_dataloader, desc = 'segment portrait'):
            imgs_ori = data_test['image_ori']
            inputs_test = data_test['image_input']
            imgs_name = data_test['img_name']
            inputs_test = inputs_test.float().cuda()

            with torch.inference_mode():
                d1,d2,d3,d4,d5,d6,d7= self.model(inputs_test)
                # normalization
                pred = d1[:,0,:,:]
                pred = normPRED(pred)
                ori_h, ori_w = imgs_ori.shape[1:3]
                imgs_Resizer = transforms.Resize((ori_h, ori_w), antialias=None)
                pred_ori = imgs_Resizer(pred).unsqueeze(-1)
                pred_ori[pred_ori<.7] = 0.
                pred_np = pred_ori.detach().cpu().numpy()

            ### find largest connected component
            pred_np[pred_np >= .7] = 1.
            pred_np = pred_np.astype(np.uint8)
            for i in range(pred_np.shape[0]):
                pred_np[i, :, :, 0] = largest_connected(pred_np[i])
            pred_ori = torch.from_numpy(pred_np).cuda().float()
            pred_np = pred_np.astype(np.float32)

            #### Maybe TODO: refine matting with https://github.com/nowsyn/InstMatt

            for i in range(len(imgs_name)):
                cv2.imwrite(os.path.join(save_folder, imgs_name[i][:-4]+'.png'), (pred_np[i,:,:,0]*255.).astype(np.uint8))

            if debug_dir != 'none':
                imgs_ori = imgs_ori.cuda()
                bg_col = torch.tensor((255, 0, 0)).cuda().float().view(1, 1, 1, -1)
                imgs_debug = imgs_ori * pred_ori + bg_col * (1.-pred_ori)
                imgs_debug = imgs_debug.detach().byte().cpu().numpy()
                if video_out is None:
                    video_out = (ffmpeg.input('pipe:0', format='rawvideo', pix_fmt='rgb24', s=str(ori_w) + 'x' + str(ori_h)).output(os.path.join(debug_dir, 'segs.mp4'), pix_fmt='yuv420p', vcodec='libx264', r=25).overwrite_output().run_async(pipe_stdin=True))
                for i in range(imgs_debug.shape[0]):
                    video_out.stdin.write(np.array(imgs_debug[i]).tobytes())

            del d1,d2,d3,d4,d5,d6,d7

        if video_out is not None:
            video_out.stdin.close()
            video_out.wait()