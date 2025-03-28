from portrait_magic.dmm_models import FLAME, flame_config
from portrait_magic.dmm_models import SMPLX, smplx_model_path
from pytorch3d.io import load_obj, save_obj
import os
import numpy as np

flame_model = FLAME(flame_config)
flame_tris = flame_model.tris[0].long().numpy()

verts, faces, aux = load_obj(os.path.join(smplx_model_path, 'smplx_teeth.obj'))
smplx_tris = faces.verts_idx.long().numpy()

sel_flame_tris = np.loadtxt('flame_wflw_ids.txt', dtype=np.int64).reshape(-1)[-10:]
sel_flame_barycoords = np.loadtxt('flame_wflw_wts.txt', dtype=np.float32)[-10:]
sel_flame_barycoords = np.concatenate((sel_flame_barycoords, 1. - np.sum(sel_flame_barycoords, axis=1, keepdims=True)), axis=1)

flame_to_smplx_ids = np.load('portrait_magic/dmm_models/smplx/SMPLX2020/SMPL-X__FLAME_vertex_ids.npy').reshape(-1).astype(np.int64)

flame_to_smplx_tris = flame_to_smplx_ids[flame_tris]
sel_smplx_tris = []
for sel_flame_tri_id in sel_flame_tris:
    flame_tri = flame_to_smplx_tris[sel_flame_tri_id]
    for smplx_tri_id, smplx_tri in enumerate(smplx_tris):
        if np.all(smplx_tri == flame_tri):
            print(f'{sel_flame_tri_id} -> {smplx_tri_id}')
            sel_smplx_tris.append(smplx_tri_id)
            # print(flame_to_smplx_tris[sel_flame_tri_id], ' ', smplx_tri)

print(os.path.realpath(os.path.join(smplx_model_path, 'iris_faces_ids.txt')))
np.savetxt(os.path.join(smplx_model_path, 'iris_faces_ids.txt'), sel_smplx_tris, fmt='%d')
np.savetxt(os.path.join(smplx_model_path, 'iris_bary_coords.txt'), sel_flame_barycoords, fmt='%f')
