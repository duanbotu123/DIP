import cv2
import numpy as np
import os
from ..utils import crop_v2
from natsort import natsorted
from torch.utils.data import Dataset
from insightface.app import FaceAnalysis
import math
import torchvision.transforms as transforms
import torch


class Image_Dataset(Dataset):
    def __init__(self, cfg):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.number_landmarks = cfg.WFLW.NUM_POINT
        self.Transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.Transform = transforms.Compose(
            [transforms.ToTensor(), self.Transform])

        # self.app = FaceAnalysis(allowed_modules=["detection"], providers=[
        #                         "CPUExecutionProvider"], root = '/nas_ssd/yudong/torch_home/.insightface') # root = '/nas_ssd/yudong/torch_home/.insightface'
        self.app = FaceAnalysis(allowed_modules=["detection"], providers=[
                                "CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def __len__(self):
        return 10

    def crop_img(self, img, bbox):
        x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32)
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = x1 + w // 2
        cy = y1 + h // 2
        center = np.array([cx, cy])
        scale = max(math.ceil(x2) - math.floor(x1),
                    math.ceil(y2) - math.floor(y1)) / 200.0

        input, trans = crop_v2(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), center, scale * 1.15, (256, 256))
        input = self.Transform(input)

        return input, trans

    def process_img(self, imgrgb):
        img = imgrgb[:, :, ::-1]
        faces = self.app.get(img, max_num=1)
        if (faces is None) or (len(faces) == 0) or (faces[0]['det_score']<.7):
            print('no face')
            return torch.zeros((3, 256, 256), dtype=torch.float32), torch.ones((2, 3), dtype=torch.float32), torch.from_numpy(img)
        bbox = faces[0].bbox
        bbox[0] = int(bbox[0] + 0.5)
        bbox[2] = int(bbox[2] + 0.5)
        bbox[1] = int(bbox[1] + 0.5)
        bbox[3] = int(bbox[3] + 0.5)
        input, trans = self.crop_img(img, bbox)

        return input, torch.from_numpy(trans.astype(np.float32))

class Folder_Dataset(Dataset):
    def __init__(self, cfg, folder):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.number_landmarks = cfg.WFLW.NUM_POINT
        self.Transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.Transform = transforms.Compose(
            [transforms.ToTensor(), self.Transform])
        self.img_paths = natsorted([os.path.join(folder, f)
                                   for f in os.listdir(folder) if f.endswith('.jpg')])

        # self.app = FaceAnalysis(allowed_modules=["detection"], providers=[
        #                         "CPUExecutionProvider"], root = '/nas_ssd/yudong/torch_home/.insightface') # root = '/nas_ssd/yudong/torch_home/.insightface'
        self.app = FaceAnalysis(allowed_modules=["detection"], providers=[
                                "CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def __len__(self):
        return len(self.img_paths)

    def crop_img(self, img, bbox):
        x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32)
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = x1 + w // 2
        cy = y1 + h // 2
        center = np.array([cx, cy])
        scale = max(math.ceil(x2) - math.floor(x1),
                    math.ceil(y2) - math.floor(y1)) / 200.0

        input, trans = crop_v2(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), center, scale * 1.15, (256, 256))
        input = self.Transform(input)

        return input, trans

    def process_img(self, imgrgb):
        img = imgrgb[:, :, ::-1]
        faces = self.app.get(img, max_num=1)
        if (faces is None) or (len(faces) == 0) or (faces[0]['det_score']<.7):
            return torch.zeros((3, 256, 256), dtype=torch.float32), torch.ones((2, 3), dtype=torch.float32), torch.from_numpy(img), 'none'
        bbox = faces[0].bbox
        bbox[0] = int(bbox[0] + 0.5)
        bbox[2] = int(bbox[2] + 0.5)
        bbox[1] = int(bbox[1] + 0.5)
        bbox[3] = int(bbox[3] + 0.5)
        input, trans = self.crop_img(img, bbox)

        return input, torch.from_numpy(trans.astype(np.float32))
        

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        faces = self.app.get(img, max_num=1)

        if (faces is None) or (len(faces) == 0) or (faces[0]['det_score']<.7):
            return torch.zeros((3, 256, 256), dtype=torch.float32), torch.ones((2, 3), dtype=torch.float32), torch.from_numpy(img), 'none'

        bbox = faces[0].bbox
        bbox[0] = int(bbox[0] + 0.5)
        bbox[2] = int(bbox[2] + 0.5)
        bbox[1] = int(bbox[1] + 0.5)
        bbox[3] = int(bbox[3] + 0.5)
        input, trans = self.crop_img(img, bbox)

        return input, torch.from_numpy(trans.astype(np.float32)), torch.from_numpy(img), os.path.basename(self.img_paths[idx])
