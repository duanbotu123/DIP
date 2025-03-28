import os
import torch
import torch.nn as nn
from .backbone import Arcface, Generator
from .geo_transform import estimate_transform_id
from kornia.geometry.transform import warp_affine

class MICAModel(nn.Module):
    """
    Shape module from MICA to extract identity feature.
    """

    def __init__(self):
        super().__init__()
        self.crop_size = 112
        # self.backbone = iresnet50(False, fp16=False)
        self.backbone = Arcface()
        self.shape_regressor = Generator()

        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        pretrained_path = os.path.join(file_dir_path, 'mica.tar')

        checkpoint = torch.load(pretrained_path, map_location='cpu')
        assert('arcface' in checkpoint)
        self.backbone.load_state_dict(checkpoint['arcface'])
        assert('flameModel' in checkpoint)
        self.shape_regressor.load_state_dict(checkpoint['flameModel'], strict=False)
        

    def forward(self, imgs, lms, lms_mode = '300w'):
        # imgs: (b, 3, h, w), lms: (b, l, 2)
        trans_mat = estimate_transform_id(lms, lms_mode=lms_mode, crop_size=self.crop_size)
        warped_imgs = warp_affine(
            imgs, trans_mat, (self.crop_size, self.crop_size), align_corners=False)
        arcface_feature = nn.functional.normalize(self.backbone(2.*warped_imgs-1.))
        return self.shape_regressor(arcface_feature)
    
    def infer_img(self, cvimg, lms, lms_mode = '300w'):
        # img: (h, w, 3), lms: (l, 2)
        img = torch.as_tensor(cvimg).cuda()[:, :, [2,1,0]].permute(2, 0, 1).unsqueeze(0).float()/255.
        lms = torch.as_tensor(lms).cuda().unsqueeze(0)
        return self.forward(img, lms, lms_mode=lms_mode)
    