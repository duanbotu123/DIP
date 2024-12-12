import mesh_to_sdf
import torch
import smplx
import trimesh
import numpy as np
from Losses import Losses
import process_mesh as pm
# import chamfer3D.dist_chamfer_3D, fscore
# from libraries.torch_mesh_isect.mesh_intersection.bvh_search_tree import BVH


'''adjust the person and object poses of one frame'''
def adjust_one_frame(o_mesh, h_para, rotation, trans, smplx_model, max_collisions=8):
    # R = torch.eye(3, requires_grad=True)  # 3x3 identity rotation matrix
    # T = torch.zeros(3, requires_grad=True)  # Translation vector
    # R[:, :] = rotation[:, :]
    # T[:, :] = trans[:, :]

    #$ set parameters, no deep copy:
    rotation.requires_grad = True
    trans.requires_grad = True
    lr = 1e-2
    optimizer = torch.optim.Adam(pm.poses2para(h_para)+[rotation, trans], lr=lr)

    #$ create meshes
    device = torch.device('cuda')
    
    # print('Number of collisions = ', collisions.shape[0])

    # col_loss = collisions.shape[0] / float(triangles.shape[1]) #$ Percentage of collisions (%)
    
    #$ add loss and optimize
    optim_iter = 1
    for i in range(optim_iter):
        h_mesh = smplx_model(betas=h_para['betas'], body_pose=h_para['body_pose'], global_orient=h_para['global_orient'],
                            transl=h_para['transl'], expression=h_para['expression'],
                            left_hand_pose=h_para['left_hand_pose'], right_hand_pose=h_para['right_hand_pose'],
                            jaw_pose=h_para['jaw_pose'])
    
        '''
        merged_mesh = trimesh.util.concatenate([o_mesh, h_mesh])
        m = BVH(max_collisions=max_collisions)
        vertices = torch.tensor(merged_mesh.vertices,
                                dtype=torch.float32, device=device)
        faces = torch.tensor(merged_mesh.faces.astype(np.int64),
                            dtype=torch.long,
                            device=device)
        triangles = vertices[faces].unsqueeze(dim=0)
        torch.cuda.synchronize()
        outputs = m(triangles).detach().cpu().numpy().squeeze()
        collisions = outputs[outputs[:, 0] >= 0, :]
        '''

        #$ calculat the SDF loss
        #$ sample points on human mesh
        #$ object mesh to sdf
        osdf = mesh_to_sdf.mesh_to_voxels(o_mesh, voxel_resolution=64, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, pad=False, check_result=False, return_gradients=False)
        from skimage.measure import marching_cubes_lewiner
        vertices, faces, _, _ = marching_cubes_lewiner(osdf, level=0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(f'sdf_mesh.ply')  # Save as PLY
        #$ add RT to the sdfs
        #$ find sdfs of these points
        #$ calculate and print the loss





        collisions_loss = collisions.shape[0] / float(triangles.shape[1])
        pene_loss = 1.
        sdf_loss = 1.
        loss = 0.1 * collisions_loss +\
               0. * pene_loss +\
               0. * sdf_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(i)

    #$ check the answer

if __name__ == '__main__':
    o_mesh = trimesh.load('/home/juyonggroup/llc0/4DGSHOI/myhoi/meshes/yogamat.ply')
    file_path = '/home/juyonggroup/llc0/4DGSHOI/myhoi/poses/smpl_full'
    smplx_poses = pm.read_smplx(file_path, [191])
    R = torch.eye(3, requires_grad=True)  # 3x3 identity rotation matrix
    T = torch.zeros(3, requires_grad=True)  # Translation vector
    model_path = '/home/juyonggroup/llc0/4DGSHOI/myhoi/libraries/'
    smplx_model = smplx.create(model_path, model_type='smplx',
                            gender='male',  # or 'female' or 'neutral'
                            ext='pkl',
                            use_pca=False,  # Set to False if using full pose parameters
                            batch_size=1)
    
    adjust_one_frame(o_mesh=o_mesh, h_para=smplx_poses, rotation=R, trans=T, smplx_model=smplx_model)
    print('3')