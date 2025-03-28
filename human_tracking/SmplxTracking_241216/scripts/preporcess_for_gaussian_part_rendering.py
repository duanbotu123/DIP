"""script for generating all files used for part uv smplx rendering
   - you need to first select subregion in blender and export {part}_sel.txt containing selected vertices indices under portrait_magic/dmm_models/adnerf_rendering/
"""
from portrait_magic.dmm_models import SMPLX, smplx_model_path
from portrait_magic.render_utils import MeshRenderer
from pytorch3d.io import load_obj, save_obj, save_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
import numpy as np
import torch
import os
import open3d as o3d
import cv2

def normalize_uvs(uvs):
    ### make uvs 0.01-0.99
    min_n, max_n = 0.01, 0.99
    bb_size = torch.max(torch.max(uvs, dim=0)[0] - torch.min(uvs, dim=0)[0]).item()
    return (uvs - torch.min(uvs, dim=0, keepdim=True)[0]) / bb_size*(max_n-min_n) + min_n

def save_obj_plain(file_path, verts, faces, uvs, uv_faces):
    # Write the OBJ file
    with open(file_path, 'w') as file:
        # Write vertices
        for v in verts:
            file.write(f'v {v[0]} {v[1]} {v[2]}\n')
        # Write UV coordinates
        for uv in uvs:
            file.write(f'vt {uv[0]} {uv[1]}\n')
        # Write faces with UV
        for i, f in enumerate(faces):
            uv_f = uv_faces[i]
            file.write(f'f {f[0]+1}/{uv_f[0]+1} {f[1]+1}/{uv_f[1]+1} {f[2]+1}/{uv_f[2]+1}\n')
        print(f'Mesh saved successfully at: {file_path}')


files_path = 'portrait_magic/dmm_models/adnerf_rendering'
part = 'wholebody'
teeth_faces_num = 36

smplx_model = SMPLX(smplx_model_path)
verts, faces, aux = load_obj(os.path.join(smplx_model_path, 'smplx_teeth.obj'))
tris = faces.verts_idx

### extract part mesh and UV it
render_part_ids = np.loadtxt(os.path.join(files_path, part + '_sel.txt'), dtype=np.int64)
verts_part = verts[render_part_ids]
old_verts_new_ids = np.ones(verts.shape[0], dtype=np.int64) - 2
for i in range(render_part_ids.shape[0]):
    old_verts_new_ids[render_part_ids[i]] = i
old_tris_new_ids = old_verts_new_ids[tris]
valid_sum = np.sum((old_tris_new_ids >= 0).astype(np.float32), axis=1)
new_tris = torch.from_numpy(old_tris_new_ids[valid_sum>2.9])
# save_obj(os.path.join(files_path, part + '.obj'), verts_part, new_tris[:-teeth_faces_num]) 
save_ply(os.path.join(files_path, part + '.ply'), verts_part, new_tris[:-teeth_faces_num]) ### since an issue with open3d, we need ply
torch.save({'upper_vid': old_verts_new_ids[8995], 'downer_vid': old_verts_new_ids[8750], 'hair_vid': old_verts_new_ids[8942]}, os.path.join(files_path, part + '_tris_info.pt'))

mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(os.path.join(files_path, part + '.ply')))
mesh.compute_uvatlas()
o3d.io.write_triangle_mesh(os.path.join(files_path, part + '_uv.obj'), mesh.to_legacy())
os.remove(os.path.join(files_path, part + '_uv.mtl'))

#### merge part_uv.obj with teeth_uv.obj
verts1, faces1, aux1 = load_obj(os.path.join(files_path, part + '_uv.obj'))
verts2, faces2, aux2 = load_obj(os.path.join(files_path, 'teeth_uv.obj'))
# Vertices combine, mesh_part contain mesh_teeth verts
vertices_com = verts1
faces_com = new_tris
# Concatenate and adjust UV coords
part_uvs = aux1.verts_uvs
teeth_uvs = aux2.verts_uvs
part_uvs = normalize_uvs(part_uvs)
teeth_uvs = normalize_uvs(teeth_uvs)
part_uv_size, teeth_uv_size = .9, .1
part_uvs *= part_uv_size
teeth_uvs = teeth_uvs*teeth_uv_size + part_uv_size
uvs_com = torch.cat([part_uvs, teeth_uvs], dim=0)
# Adjusting UV face indices for the second mesh
uv_faces1 = faces1.textures_idx
uv_faces2 = faces2.textures_idx + part_uvs.shape[0]  # Offset by the number of UV vertices in the first mesh
uv_faces_com = torch.cat([uv_faces1, uv_faces2])
file_path = os.path.join(files_path, f'{part}_with_teeth_uv.obj')
save_obj_plain(file_path, vertices_com, faces_com, uvs_com, uv_faces_com)

verts, faces, aux = load_obj(os.path.join(files_path, f'{part}_with_teeth_uv.obj'))
uv_coords = aux.verts_uvs.float().cuda()
uv_coords[..., 1] = 1. - uv_coords[..., 1]
tris_uv = faces.textures_idx.cuda().int()
tris_verts = faces.verts_idx.cuda().long()
verts = verts.cuda().float()

mesh_renderer = MeshRenderer()

uv_size = 768
rast_out = mesh_renderer.rasterize_uv_img(uv_coords, tris_uv, uv_size)
print(rast_out.shape, float(torch.sum((rast_out[..., 3]>0).float())) / uv_size / uv_size)
faces_num = tris_uv.shape[0]

##### save info, upper_teeth: 1164, downer_teeth: 1049
rast_out[..., 3] -= 1
smplx_part = rast_out[0, :, :, :][(rast_out[0, :, :, 3] > -.5)].reshape(-1, 4)

valid_tris_verts = tris_verts[smplx_part[:, 3].long()]
valid_verts = verts[valid_tris_verts[:,0]] * smplx_part[:, 0:1] + verts[valid_tris_verts[:,1]] * smplx_part[:, 1:2] + verts[valid_tris_verts[:,2]] * (1. - torch.sum(smplx_part[:, 0:2], dim=-1, keepdim=True))

valid_img_indices = torch.nonzero(rast_out[0, :, :, 3] > -.5, as_tuple=True)
img_valids = (torch.zeros_like(rast_out[0, :, :, 3]).long() - 1).cpu()
img_valids[valid_img_indices[0], valid_img_indices[1]] = torch.arange(valid_img_indices[0].shape[0])

neibor_indices = []
mindis = 2e-2
valid_img_ys = []
valid_img_xs = []
center_ids = []
neighbor_nums = []
for y in range(1, img_valids.shape[0]-1):
    for x in range(1, img_valids.shape[1]-1):
        if img_valids[y,x] == -1:
            continue
        valid_img_ys.append(y)
        valid_img_xs.append(x)
        neibor_ids = []
        center_idx = img_valids[y,x]
        neibor_ids.append(center_idx)
        center_ids.append(center_idx)
        neighbor_num = 0
        for dy in {-1, 1}:
            for dx in {-1, 1}:
                cur_idx = img_valids[y+dy, x+dx]
                if cur_idx == -1:
                    neibor_ids.append(center_idx)
                else:
                    neibor_ids.append(cur_idx)
                    neighbor_num += 1
        # if neighbor_num < 2:
        #     print(neighbor_num, y, x)
        neighbor_nums.append(neighbor_num)
        neibor_indices.append(neibor_ids)

valid_img_ys = np.array(valid_img_ys, dtype=np.int64)
valid_img_xs = np.array(valid_img_xs, dtype=np.int64)
center_ids = np.array(center_ids, dtype=np.int64) - np.arange(len(center_ids))
neighbor_nums = np.array(neighbor_nums, dtype=np.int64)
unique, counts = np.unique(neighbor_nums, return_counts=True)
print(np.asarray((unique, counts)).T)

print(np.max(center_ids), np.min(center_ids))

neibor_indices = np.array(neibor_indices)
print(neibor_indices.shape, (img_valids.shape[0]-2) * (img_valids.shape[1]-2), neibor_indices.shape[0] / ((img_valids.shape[0]) * (img_valids.shape[1])))

is_teeth_tris = (smplx_part[:, 3] >= (faces_num-teeth_faces_num))
is_skin_tris = (smplx_part[:, 3] < (faces_num-teeth_faces_num))

mid_y = torch.median(valid_verts[is_teeth_tris][:, 1]).item()
is_upper = (valid_verts[:, 1] >= mid_y)
is_teeth_upper_tris = is_teeth_tris & is_upper
is_teeth_downer_tris = is_teeth_tris & (~is_upper)

# uv_mask_img = ((rast_out[0, :, :, 3] > 0).float()*255).byte().detach().cpu().numpy()
# uv_mask_img[valid_img_ys[is_skin_tris.cpu()], valid_img_xs[is_skin_tris.cpu()]] = 50
# uv_mask_img[valid_img_ys[is_teeth_upper_tris.cpu()], valid_img_xs[is_teeth_upper_tris.cpu()]] = 100
# uv_mask_img[valid_img_ys[is_teeth_downer_tris.cpu()], valid_img_xs[is_teeth_downer_tris.cpu()]] = 200
# for y in range(1, img_valids.shape[0]-1):
#     for x in range(1, img_valids.shape[1]-1):
#         if img_valids[y,x] == -1:
#             continue
#         neighbor_num = 0
#         for dy in {-1, 1}:
#             for dx in {-1, 1}:
#                 cur_idx = img_valids[y+dy, x+dx]
#                 if cur_idx == -1:
#                     continue
#                 else:
#                     neighbor_num += 1
#         if neighbor_num <= 2:
#             uv_mask_img[y, x] = 150
# cv2.imwrite('uv_mask.jpg', uv_mask_img)


tris_info_dict = torch.load(os.path.join(files_path, part + '_tris_info.pt'))
tris_info_dict['smplx_tris'] = smplx_part[is_skin_tris][..., [3,0,1]].cpu()
# tris_info_dict['teeth_upper_tris'] = smplx_part[is_teeth_upper_tris][..., [3,0,1]].cpu()
# tris_info_dict['teeth_downer_tris'] = smplx_part[is_teeth_downer_tris][..., [3,0,1]].cpu()
tris_info_dict['teeth_tris'] = smplx_part[is_teeth_tris][..., [3,0,1]].cpu()

tris_info_dict['gs_tris'] = smplx_part[..., [3,0,1]].cpu()
tris_info_dict['upper_teeth_inds'] = torch.nonzero(is_teeth_upper_tris).cpu()
tris_info_dict['downer_teeth_inds'] = torch.nonzero(is_teeth_downer_tris).cpu()

print(tris_info_dict['upper_teeth_inds'].shape, tris_info_dict['upper_teeth_inds'].dtype, tris_info_dict['downer_teeth_inds'].shape)

tris_info_dict['lap_indices'] = torch.from_numpy(neibor_indices).long()
torch.save(tris_info_dict, os.path.join(files_path, part + '_tris_info.pt'))
