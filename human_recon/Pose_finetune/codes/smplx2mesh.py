import torch
import smplx
import numpy as np
import trimesh

# Path to the SMPL-X model file (download from https://smpl-x.is.tue.mpg.de/)
model_path = '/data_b/llc/4DGSGEHOI/myhoi/SMPL/'

# Create the SMPL-X model instance
print('Create the SMPL-X model instance')
smplx_model = smplx.create(model_path, model_type='smplx',
                           gender='male',  # or 'female' or 'neutral'
                           ext='pkl',
                           use_pca=False,  # Set to False if using full pose parameters
                           batch_size=1)

poses = np.load('/data_b/llc/4DGSGEHOI/AnimatableGaussians/poses/thuman4/pose_mocap100.npz')
# print(poses.files) #$ print the keys

# Example SMPL-X parameters (you can replace with your own parameters)
'''# Shape parameters (betas), Pose parameters (body_pose, global_orient, etc.)
betas = torch.zeros([1, 10])  # Shape coefficients (10 shape parameters)
body_pose = torch.zeros([1, 63])  # Body pose coefficients
global_orient = torch.zeros([1, 3])  # Global orientation
transl = torch.zeros([1, 3])  # Translation
expression = torch.zeros([1, 10])  # Expression (only for SMPL-X)
left_hand_pose = torch.zeros([1, 45])  # Left hand pose
right_hand_pose = torch.zeros([1, 45])  # Right hand pose
jaw_pose = torch.zeros([1, 3])  # Jaw pose (only for SMPL-X)
'''
betas = torch.from_numpy(poses['betas'])  # Shape coefficients (10 shape parameters)
for i in range(48, 49):
    body_pose = torch.from_numpy(poses['body_pose'])[i:i+1]  # Body pose coefficients
    global_orient = torch.from_numpy(poses['global_orient'])[i:i+1]  # Global orientation
    transl = torch.from_numpy(poses['transl'])[i:i+1]  # Translation
    expression = torch.from_numpy(poses['expression'])[i:i+1]  # Expression (only for SMPL-X)
    left_hand_pose = torch.from_numpy(poses['left_hand_pose'])[i:i+1]  # Left hand pose
    right_hand_pose = torch.from_numpy(poses['right_hand_pose'])[i:i+1]  # Right hand pose
    jaw_pose = torch.from_numpy(poses['jaw_pose'])[i:i+1]  # Jaw pose (only for SMPL-X)

    # print('Shapes: ', global_orient.shape)

    # Forward pass through the SMPL-X model to get the output mesh vertices
    output = smplx_model(betas=betas, body_pose=body_pose, global_orient=global_orient,
                        transl=transl, expression=expression,
                        left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                        jaw_pose=jaw_pose)

    # The output vertices of the mesh
    vertices = output.vertices.detach().cpu().numpy().squeeze()

    # The faces of the SMPL-X mesh (static, part of the model)
    faces = smplx_model.faces

    # Convert to Trimesh format and save
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(f'smplx_mesh_{i}.ply')  # Save as PLY
    # mesh.export('smplx_mesh.obj')  # Save as OBJ

    print(f"Mesh saved as 'smplx_mesh_{i}.ply'")
