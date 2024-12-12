import torch
import smplx
import trimesh
import numpy as np
import json
import os

'''This file constains functions for processing meshes, especially smplx.'''
def read_smplx(data_path, list):
    with open(os.path.join(data_path, '000000.json'), 'r') as file:
        poses = json.load(file)[0]
        # print(poses.files)
    betas = torch.tensor(poses['shapes'])  # Shape coefficients (10 shape parameters)
    smplx_poses = {}
    smplx_poses['betas'] = betas
    for i in list:
        with open(os.path.join(data_path, f'000{i:03d}.json'), 'r') as file:
            poses = json.load(file)[0]
        body_pose = torch.tensor(poses['poses'])[:, 3:66]  # Body pose coefficients
        global_orient = torch.tensor(poses['Rh'])  # Global orientation
        transl = torch.tensor(poses['Th'])  # Translation
        expression = torch.tensor(poses['expression'])  # Expression (only for SMPL-X)
        left_hand_pose = torch.tensor(poses['poses'])[:, 75:120]  # Left hand pose
        right_hand_pose = torch.tensor(poses['poses'])[:, 120:]  # Right hand pose
        jaw_pose = torch.tensor(poses['poses'])[:, 66:69]  # Jaw pose (only for SMPL-X)

        # update pose
        if not len(smplx_poses)-1:
            smplx_poses['body_pose'] = body_pose
            smplx_poses['global_orient'] = global_orient
            smplx_poses['transl'] = transl
            smplx_poses['expression'] = expression
            smplx_poses['left_hand_pose'] = left_hand_pose
            smplx_poses['right_hand_pose'] = right_hand_pose
            smplx_poses['jaw_pose'] = jaw_pose
        else:
            smplx_poses['body_pose'] = torch.cat([smplx_poses['body_pose'], body_pose], dim = 0)
            smplx_poses['global_orient'] = torch.cat([smplx_poses['global_orient'], global_orient], dim = 0)
            smplx_poses['transl'] = torch.cat([smplx_poses['transl'], transl], dim = 0)
            smplx_poses['expression'] = torch.cat([smplx_poses['expression'], expression], dim = 0)
            smplx_poses['left_hand_pose'] = torch.cat([smplx_poses['left_hand_pose'], left_hand_pose], dim = 0)
            smplx_poses['right_hand_pose'] = torch.cat([smplx_poses['right_hand_pose'], right_hand_pose], dim = 0)
            smplx_poses['jaw_pose'] = torch.cat([smplx_poses['jaw_pose'], jaw_pose], dim = 0)
        
    # for a, b in smplx_poses.items():
    #     print(a, b.shape)
        
    return smplx_poses

def poses2para(smplx_poses):
    smplx_poses['body_pose'].requires_grad = True
    smplx_poses['global_orient'].requires_grad = True
    smplx_poses['transl'].requires_grad = True
    smplx_poses['expression'].requires_grad = True
    smplx_poses['left_hand_pose'].requires_grad = True
    smplx_poses['right_hand_pose'].requires_grad = True
    smplx_poses['jaw_pose'].requires_grad = True
    param_list = [smplx_poses['body_pose'], smplx_poses['global_orient'], smplx_poses['transl'], smplx_poses['expression'], 
                  smplx_poses['left_hand_pose'], smplx_poses['right_hand_pose'], smplx_poses['jaw_pose']]
    return param_list

def smplx2mesh(smplx_model, smplx_poses, ind_list, output_path):
    for i in ind_list:

        # Forward pass through the SMPL-X model to get the output mesh vertices
        output = smplx_model(betas=smplx_poses['betas'], body_pose=smplx_poses['body_pose'][i].unsqueeze(0), global_orient=smplx_poses['global_orient'][i].unsqueeze(0),
                            transl=smplx_poses['transl'][i].unsqueeze(0), expression=smplx_poses['expression'][i].unsqueeze(0),
                            left_hand_pose=smplx_poses['left_hand_pose'][i].unsqueeze(0), right_hand_pose=smplx_poses['right_hand_pose'][i].unsqueeze(0),
                            jaw_pose=smplx_poses['jaw_pose'][i].unsqueeze(0))

        # The output vertices of the mesh
        vertices = output.vertices.detach().cpu().numpy().squeeze()

        # The faces of the SMPL-X mesh (static, part of the model)
        faces = smplx_model.faces

        # Convert to Trimesh format and save
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(output_path + f'smplx_mesh_{i}.ply')  # Save as PLY
        # mesh.export('smplx_mesh.obj')  # Save as OBJ

        print(f"Mesh saved as ", output_path + f'smplx_mesh_{i}.ply')