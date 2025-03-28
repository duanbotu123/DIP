from portrait_magic.dmm_models import FLAME, flame_config
from portrait_magic.face_landmarks import DSLPT_utils
from portrait_magic.wb_landmarks import DWpose
from portrait_magic.render_utils import MeshRenderer
from portrait_magic.render_utils.geometry_utils import forward_rott
from tqdm import tqdm
import numpy as np
import torch
import cv2
import os

def draw_landmark(landmark, image):
    for (x, y) in (landmark + 0.5).astype(np.int32):
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    return image

if __name__ == "__main__":

    tmp_dir = 'tmp/ldmks'
    os.makedirs(tmp_dir, exist_ok=True)

    h, w = 512, 512
    img_size = [h, w]
    fx, fy, cx, cy = 1800., 1800., w/2., h/2.
    shape_dim, exp_dim, tex_dim = flame_config.n_shape, flame_config.n_exp, flame_config.n_tex

    flame_model = FLAME(flame_config).cuda()
    ldmk_detector = DSLPT_utils()
    renderer = MeshRenderer()
    
    
    cam_para = torch.tensor((fx, fy, cx, cy), dtype=torch.float32).cuda().unsqueeze(0)
    trans = torch.tensor((0, 0, -1.5)).float().cuda().view(1, 3)
    euler_angle = torch.tensor((0,0,0)).float().cuda().view(1, 3)
    gamma = torch.zeros((1, 27)).float().cuda()
    
    rendered_num = 100
    batch_tri_ids = []
    batch_bary_coords = []

    torch.set_grad_enabled(False)


    ### filter out backface tris
    back_verts = np.loadtxt('scripts/backface.txt', dtype=np.int64)
    is_back = np.zeros(5023, dtype=np.int64)
    is_back[back_verts] = 1
    is_tris_back = is_back[flame_model.tris[0].long().cpu()]
    is_tris_sum = np.sum(is_tris_back, axis=1)
    valid_tris_inds = is_tris_sum<2.9
    new_tris_old_ids = np.zeros(valid_tris_inds.shape[0], dtype=np.int64) - 1
    cur_id = 0
    for i in range(valid_tris_inds.shape[0]):
        if is_tris_sum[i]<2.9:
            new_tris_old_ids[cur_id] = i
            cur_id += 1

    for i in tqdm(range(rendered_num)):
        shape_para = torch.randn((1, shape_dim)).float().cuda()*.8
        exp_para = torch.randn((1, exp_dim)).float().cuda()*.5
        tex_para = torch.randn((1, tex_dim)).float().cuda()*.5
        jaw_para = torch.zeros_like(trans)
        jaw_para[0, 0] += np.random.uniform(0.10, 0.30)
        vertices = flame_model.forward_geo(shape_para, exp_para, jaw_pose_params = jaw_para)
        vertices_cam = forward_rott(vertices, euler_angle, trans)
        texture = flame_model.forward_tex(tex_para)
        lights = torch.rand(1,27).float().cuda()/20.

        color_img, rast_out = renderer.forward_rendering_uv(vertices_cam, flame_model.tris[:, valid_tris_inds], texture, flame_model.uv_coords, flame_model.tris_uv[:, valid_tris_inds], cam_para, img_size, lights=lights, return_rast_out=True)
        mask = (rast_out[..., 3:] > 0).float()
        color_img = color_img*mask + (1.-mask)
        
        img_rgb = torch.clamp(color_img[0]*255, 0, 255).detach().byte().cpu().numpy()
        ldmks = ldmk_detector.run_image(img_rgb)
        landmark = ldmks[:, :2]
        img_rgb = draw_landmark(landmark, img_rgb)
        cv2.imwrite(os.path.join(tmp_dir, str(i).zfill(4) + '.jpg'), img_rgb[:, :, ::-1])   

        tri_ids = rast_out[0, (landmark[:, 1]+.5).astype(np.int64), (landmark[:, 0]+.5).astype(np.int64), 3].int()
        bary_coords = rast_out[0, (landmark[:, 1]+.5).astype(np.int64), (landmark[:, 0]+.5).astype(np.int64), :2]
        batch_tri_ids.append(tri_ids)
        batch_bary_coords.append(bary_coords)

    batch_tri_ids = torch.stack(batch_tri_ids)
    batch_bary_coords = torch.stack(batch_bary_coords)

    most_tri_ids = []
    for i in range(batch_tri_ids.shape[1]):
        most_tri_ids.append(torch.mode(batch_tri_ids[:, i:i+1][batch_tri_ids[:, i:i+1] > 0], dim=0, keepdim=True)[0])
    most_tri_ids = torch.cat(most_tri_ids, dim=-1)

    # print((batch_tri_ids == most_tri_ids).shape)
    print(torch.sum((batch_tri_ids == most_tri_ids).float(), dim=0)/rendered_num)
    valid_mask = (batch_tri_ids == most_tri_ids).float().unsqueeze(-1)
    mean_coords = torch.sum(batch_bary_coords*valid_mask, dim=0, keepdim=True)/torch.sum(valid_mask, dim=0, keepdim=True)
    print(most_tri_ids)

    ldmks_tri_ids = most_tri_ids.reshape(-1).cpu().numpy()-1
    ldmks_bary_coords = mean_coords.reshape(-1, 2).cpu().numpy()
    ldmks_bary_coords = np.concatenate((ldmks_bary_coords, 1.-np.sum(ldmks_bary_coords, axis=1, keepdims=True)), axis=1)

    save_dir = flame_config.data_dir 
    np.savetxt(os.path.join(save_dir, 'wflw_tri_ids_fix.txt'), new_tris_old_ids[ldmks_tri_ids], fmt='%d')
    np.savetxt(os.path.join(save_dir, 'wflw_bary_coords_add_fix.txt'), ldmks_bary_coords, fmt='%.6f')
    