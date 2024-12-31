import os
import torch
import numpy as np
from torch.optim import Adam
import torch.nn as nn

'''Tracking the object '''
'''Tracking the human'''

#$ define the parameters
'''pose_init_path = 'path/to/initial_pose_parameters.npz'
pose_data = np.load(pose_init_path)
pose_params = torch.tensor(pose_data['poses'], dtype=torch.float32, requires_grad=True)
'''

'''6 DoF to transformation matrix'''
def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to a rotation matrix using PyTorch."""
    R_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(roll), -torch.sin(roll)],
                        [0, torch.sin(roll), torch.cos(roll)]])
    
    R_y = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                        [0, 1, 0],
                        [-torch.sin(pitch), 0, torch.cos(pitch)]])
    
    R_z = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                        [torch.sin(yaw), torch.cos(yaw), 0],
                        [0, 0, 1]])
    
    # Combined rotation matrix
    R = torch.mm(R_z, torch.mm(R_y, R_x))
    return R

def pose_to_transformation_matrix(tx, ty, tz, roll, pitch, yaw):
    """Convert a 6 DoF pose to a transformation matrix using PyTorch."""
    # Translation vector
    t = torch.tensor([tx, ty, tz])
    
    # Rotation matrix from Euler angles
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    
    # Create the transformation matrix
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

'''Read the coarse smplx'''



Rotation = torch.tensor([0.5, 0.5, 0.5, 0.5]).to()
Trans = torch.tensor([0., 0., 0.]).to()
pose_params = [Rotation, Trans]

# Load the ground truth images
img_gt_path = 'path/to/ground_truth_images'
img_gt_files = sorted(os.listdir(img_gt_path))
img_gt = [torch.tensor(np.load(os.path.join(img_gt_path, f)), dtype=torch.float32) for f in img_gt_files]
img_gt = torch.stack(img_gt)  # Stack into a single tensor


# Define the optimizer
optimizer = Adam([pose_params], lr=0.01)

# Define the loss function (L1 loss)
loss_fn = nn.L1Loss()

# Optimization loop
num_iterations = 100  # Number of iterations for optimization
for iteration in range(num_iterations):
    optimizer.zero_grad()
    
    # Render the image with current pose parameters
    img = render(pose_params)
    
    # Calculate the loss between the rendered image and ground truth image
    loss = loss_fn(img, img_gt)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")

# Save the optimized pose parameters
optimized_pose_params = pose_params.detach().numpy()
output_path = 'path/to/optimized_pose_parameters.npz'
np.savez(output_path, poses=optimized_pose_params)

print(f"Optimized pose parameters have been saved to {output_path}")
