from portrait_magic.render_utils import MeshRenderer
from pytorch3d.io import load_obj
import numpy as np
import torch
import cv2
import os

# file_dir_path = os.path.dirname(os.path.realpath(__file__))
file_dir_path = 'portrait_magic/dmm_models/adnerf_rendering'
part = 'upper_shoulder'
_, faces, aux = load_obj(os.path.join(file_dir_path, part + '_uv.obj'))

uv_coords = aux.verts_uvs.float().cuda()
uv_coords[..., 1] = 1. - uv_coords[..., 1]
tris_uv = faces.textures_idx.cuda().int()

mesh_renderer = MeshRenderer()

uv_size = 256
rast_out = mesh_renderer.rasterize_uv_img(uv_coords, tris_uv, uv_size)
print(rast_out.shape, float(torch.sum((rast_out[..., 3]>0).float())) / uv_size / uv_size)
uv_mask_img = ((rast_out[0, :, :, 3] > 0).float()*255).byte().detach().cpu().numpy()
faces_num = tris_uv.shape[0]
teeth_faces_num = 36
uv_mask_img[((rast_out[0, :, :, 3]-1)>=(faces_num-teeth_faces_num)).cpu().numpy()] = 155
cv2.imwrite('tmp/uv_mask.jpg', uv_mask_img)

##### save info, upper_teeth: 1164, downer_teeth: 1049
rast_out[..., 3] -= 1
smplx_part = rast_out[0, :, :, :][(rast_out[0, :, :, 3] > -.5) & (rast_out[0, :, :, 3] < (faces_num-teeth_faces_num - .5))].reshape(-1, 4)
teeth_part = rast_out[0, :, :, :][(rast_out[0, :, :, 3] > (faces_num-teeth_faces_num - .5))].reshape(-1, 4)
print(smplx_part.shape, teeth_part.shape, (smplx_part.shape[0]+teeth_part.shape[0])/uv_size/uv_size)

tris_info_dict = torch.load(os.path.join(file_dir_path, part + '_tris_info.pt'))
tris_info_dict['smplx_tris'] = smplx_part[..., [3,0,1]].cpu()
tris_info_dict['teeth_tris'] = teeth_part[..., [3,0,1]].cpu()

teeth_info = teeth_part[..., [3,0,1]].cpu().numpy()
teeth_info[:, 0] -= faces_num
np.savetxt(os.path.join(file_dir_path, part + 'teeth_tris.txt'), teeth_info, fmt='%f')

torch.save(tris_info_dict, os.path.join(file_dir_path, part + '_tris_info.pt'))
