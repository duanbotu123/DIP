from portrait_magic.dmm_models import SMPLX, smplx_model_path
from pytorch3d.io import load_obj, save_obj
import numpy as np
import torch
import os
import open3d as o3d

smplx_model = SMPLX(smplx_model_path)
verts, faces, aux = load_obj(os.path.join(smplx_model_path, 'smplx_teeth.obj'))
tris = faces.verts_idx

jaw_pose = torch.zeros(1, 3)
jaw_pose[0, 0] = .1
smplx_out = smplx_model.forward(jaw_pose = jaw_pose)
verts = smplx_out.vertices[0]

# file_dir_path = os.path.dirname(os.path.realpath(__file__))
file_dir_path = 'portrait_magic/dmm_models/adnerf_rendering'
part = 'upper_shoulder'
adnerf_render_part_ids = np.loadtxt(os.path.join(file_dir_path, part + '_sel.txt'), dtype=np.int64)

verts_adnerf = verts[adnerf_render_part_ids]
old_verts_new_ids = np.ones(verts.shape[0], dtype=np.int64) - 2
for i in range(adnerf_render_part_ids.shape[0]):
    old_verts_new_ids[adnerf_render_part_ids[i]] = i
old_tris_new_ids = old_verts_new_ids[tris]
valid_sum = np.sum((old_tris_new_ids >= 0).astype(np.float32), axis=1)
new_tris = torch.from_numpy(old_tris_new_ids[valid_sum>2.9])
save_obj(os.path.join(file_dir_path, part + '_tris.obj'), verts_adnerf, new_tris)
teeth_faces_num = 36
save_obj(os.path.join(file_dir_path, part + '_no_teeth_tris.obj'), verts_adnerf, new_tris[:-teeth_faces_num])
torch.save({'upper_vid': old_verts_new_ids[8995], 'downer_vid': old_verts_new_ids[8750], 'hair_vid': old_verts_new_ids[8942]}, os.path.join(file_dir_path, part + '_tris_info.pt'))
print(old_verts_new_ids[8995], old_verts_new_ids[8750], old_verts_new_ids[8942])

mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(os.path.join(file_dir_path, part + '_tris.obj')))
mesh.compute_uvatlas()
o3d.io.write_triangle_mesh(os.path.join(file_dir_path, part + '_uv.obj'), mesh.to_legacy())

mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(os.path.join(file_dir_path, part + '_no_teeth_tris.obj')))
mesh.compute_uvatlas()
o3d.io.write_triangle_mesh(os.path.join(file_dir_path, part + '_no_teeth_uv.obj'), mesh.to_legacy())
