import torch
import os
import os.path as osp
from typing import *
from .model import MattingNetwork

class RVMWrapper():
    def __init__(self, device = 'cuda:0'):
        self.model = MattingNetwork('resnet50').eval().to(device) 
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model.load_state_dict(torch.load(osp.join(file_dir_path, 'rvm_resnet50.pth')))
        self.init_rec()
        self.downsample_ratio = 0.3
    
    def init_rec(self):
        self.rec = [None] * 4

    def run(self, src):
        with torch.no_grad():
            fgr, pha, *self.rec = self.model(src, *self.rec, self.downsample_ratio)  # Cycle the recurrent states.
            return pha        
