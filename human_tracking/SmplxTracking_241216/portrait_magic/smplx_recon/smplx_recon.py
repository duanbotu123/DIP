import os
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pytorch3d.io import load_obj
from ..dmm_models.smplx.smplx_config import smlx_config
# from .folder_loader import Folder_Dataset
from ..render_utils import MeshRenderer
from ..utils_funcs.video_writer import VideoWriter
from .AiOS.models.registry import MODULE_BUILD_FUNCS
from .AiOS.util import misc as utils
from .AiOS.util.config import DictAction, cfg
from .AiOS.datasets.INFERENCE_demo import INFERENCE_demo
from .AiOS.detrsmpl.data.datasets import build_dataloader
from .AiOS.engine import inference

def build_model_main(args, cfg):
    print(args.modelname)
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors, _ = build_func(
        args, cfg)
    return model, criterion, postprocessors, _

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector',
                                     add_help=False)
    # parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    
    # training parameters
    parser.add_argument('--output_dir',
                        default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path',
                        help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--to_vid', action='store_true')
    parser.add_argument('--inference', action='store_true')
    # distributed training parameters

    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank',
                        type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--amp',
                        action='store_true',
                        help='Train with mixed precision')

    parser.add_argument('--inference_input', default=None, type=str)
    parser.add_argument('--video_path', default=None, type=str)
    return parser

class SMPLX_RECON():
    def __init__(self):
        self.device = 'cuda:0'

        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
        args = parser.parse_args()
        args.inference = True
        args.eval = True
        args.batch_size = 8
        args.backbone="resnet50"
        args.num_person=1
        args.threshold = 0.1

        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        args.config_file = os.path.join(file_dir_path, 'AiOS/config/aios_smplx_demo.py')
        print('Loading config file from {}'.format(args.config_file))
        shutil.copy2(args.config_file, os.path.join(file_dir_path, 'AiOS/config/aios_smplx.py'))
        from .AiOS.config.config import cfg

        if args.options is not None:
            cfg.merge_from_dict(args.options)
        cfg_dict = cfg._cfg_dict.to_dict()
        args_vars = vars(args)
        for k, v in cfg_dict.items():
            if k not in args_vars:
                setattr(args, k, v)
            else:
                continue
                raise ValueError('Key {} can used by args only'.format(k))
        args.use_ema = False
        args.debug = False
        self.model, self.criterion, self.postprocessors, _ = build_model_main(args, cfg)
        self.model.to(self.device)
        
        checkpoint = torch.load(os.path.join(file_dir_path, 'AiOS/aios_checkpoint.pth'), map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        _, faces, aux = load_obj(os.path.join(smlx_config.model_path, 'smplx_uv.obj'))
        self.uv_coords = aux.verts_uvs[None, ...].float().cuda()
        self.uv_coords[..., 1] = 1. - self.uv_coords[..., 1]
        self.tris_uv = faces.textures_idx[None, ...].cuda().int()
        self.tris = faces.verts_idx[None, ...].cuda().int()

        self.render_type = 'nvdiffrast' ## 'nvdiffrast'

        if self.render_type == 'nvdiffrast':
            self.mesh_renderer = MeshRenderer().cuda()
        self.args = args
        print('bbbbbbbb')

    def run_folder(self, imgs_folder, save_folder, debug_dir='none'):
        dataset_val = eval(INFERENCE_demo)(imgs_folder, save_folder)
        data_loader_val = build_dataloader(dataset_val, self.args.batch_size, 0, dist=False, shuffle=False)


        os.environ['EVAL_FLAG'] = 'TRUE'
        inference(self.model, self.criterion, self.postprocessors, data_loader_val, self.device, save_folder, wo_class_error=True, args=self.args)   

        # folder_dataset = Folder_Dataset(imgs_folder)
        # folder_loader = torch.utils.data.DataLoader(
        #     folder_dataset, batch_size=5, shuffle=False, num_workers=5, pin_memory=True, drop_last=False)
        # video_out = None

        # saved_keys = ['smplx_body_pose', 'smplx_lhand_pose', 'smplx_rhand_pose', 'smplx_jaw_pose', 'smplx_shape', 'smplx_expr', 'smplx_root_pose', 'cam_trans', 'cam_para']
        # saved_dict = {}
        # for key in saved_keys:
        #     saved_dict[key] = []
        # saved_dict['name_pre'] = []
        # saved_dict['cam_para'] = []
        
        # for img, img_ori, bboxes, name_pre in tqdm(folder_loader, desc='osx smplx recon'):
        #     img = img.cuda().permute(0, 3, 1, 2).float() / 255.
        #     inputs = {'img': img}
        #     targets, meta_info = {}, {}
        #     with torch.inference_mode():
        #         out = self.demoer.model(inputs, targets, meta_info, 'test')

        #     # for key in out.keys():
        #     #     print(key, out[key].shape)
            
        #         for key in saved_keys:
        #             if key in out.keys():
        #                 saved_dict[key].append(out[key])
        #         saved_dict['name_pre'].extend(name_pre)

        #         for i in range(img_ori.shape[0]):
        #             bbox = bboxes[i].numpy()
        #             focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
        #             princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
        #             cam_para = torch.tensor((focal[0], focal[1], princpt[0], princpt[1])).float().cuda()[None, ...]
        #             saved_dict['cam_para'].append(cam_para)
                
        #         if debug_dir != 'none':
        #             meshes = out['smplx_mesh_cam']
        #             img_ori = img_ori.numpy()
        #             if video_out is None:
        #                 video_out = VideoWriter(os.path.join(debug_dir, 'smplx_recon.mp4'))
                        
        #             for i in range(img_ori.shape[0]):
        #                 bbox = bboxes[i].numpy()
        #                 vis_img = np.array(img_ori[i])
        #                 focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
        #                 princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
                        
        #                 vertices_cam = meshes[i:i+1]
        #                 vertices_cam[..., 1:] *= -1.
        #                 cam_para = torch.tensor((focal[0], focal[1], princpt[0], princpt[1])).float().cuda()[None, ...]
        #                 render_vis = self.mesh_renderer.forward_visualization_geo(vertices_cam, self.tris, cam_para, vis_img.shape[:2], torch.as_tensor(vis_img).cuda()[None, ...].float()/255.)
        #                 vis_img = render_vis[0]  
        #                 vis_img = vis_img.astype(np.uint8)

        #                 video_out.write_frame(vis_img)
        
        # for key in saved_keys:
        #     saved_dict[key] = torch.cat(saved_dict[key], dim=0).detach().cpu()
        # torch.save(saved_dict, os.path.join(save_folder, 'osx_recon.pth'))

        # if video_out is not None:
        #     video_out.close()
            