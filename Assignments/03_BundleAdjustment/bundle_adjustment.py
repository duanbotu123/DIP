"""
Bundle Adjustment Implementation using PyTorch
Assignment 3 - DIP Course

Recover 3D point coordinates, camera extrinsics (R, T), and focal length
from 2D observations across 50 views using gradient-based optimization.

Usage:
    conda activate animategauss
    python bundle_adjustment.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch3d.transforms import euler_angles_to_matrix

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_VIEWS = 50
NUM_POINTS = 20000
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
CX = IMAGE_WIDTH / 2.0   # 512.0
CY = IMAGE_HEIGHT / 2.0  # 512.0

# Optimization hyperparameters
NUM_ITERATIONS = 3000
LR_POINTS = 0.05
LR_CAMERA = 0.005
LR_FOCAL = 0.001

# Initialization
INIT_DISTANCE = 2.5      # camera distance from origin along -Z
INIT_FOV_DEG = 60.0      # initial field of view estimate

# GPU selection
GPU_ID = 0

# ============================================================
# Data Loading
# ============================================================
def load_data():
    """Load 2D observations and point colors."""
    points2d = np.load(os.path.join(DATA_DIR, "points2d.npz"))
    colors = np.load(os.path.join(DATA_DIR, "points3d_colors.npy"))
    return points2d, colors


def prepare_observation_tensors(points2d_npz, device):
    """
    Convert observations to torch tensors.
    Returns:
        obs_2d: (N_views, N_points, 2) - pixel coordinates
        vis_mask: (N_views, N_points) - visibility mask (bool)
    """
    obs_list = []
    vis_list = []
    for i in range(NUM_VIEWS):
        key = f"view_{i:03d}"
        data = points2d_npz[key]  # (20000, 3): [x, y, visibility]
        obs_list.append(data[:, :2])     # (N, 2)
        vis_list.append(data[:, 2])      # (N,)

    obs_2d = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=device)
    vis_mask = torch.tensor(np.stack(vis_list, axis=0), dtype=torch.float32, device=device)
    return obs_2d, vis_mask


# ============================================================
# Projection Function
# ============================================================
def project_points(points3d, euler_angles, translations, focal):
    """
    Project 3D points to 2D pixel coordinates.

    Args:
        points3d: (N_points, 3) - 3D point coordinates
        euler_angles: (N_views, 3) - Euler angles for each camera (XYZ convention)
        translations: (N_views, 3) - Translation vectors for each camera
        focal: scalar - focal length (shared across all cameras)

    Returns:
        projected: (N_views, N_points, 2) - predicted 2D pixel coordinates
    """
    # Convert Euler angles to rotation matrices: (N_views, 3, 3)
    R = euler_angles_to_matrix(euler_angles, convention="XYZ")

    # Transform points to camera coordinates:
    # Xc = R @ P^T + T  =>  (N_views, 3, N_points)
    # points3d: (N_points, 3) -> (1, 3, N_points)
    P = points3d.t().unsqueeze(0)  # (1, 3, N_points)

    # Xc = R @ P + T  where R: (N_views, 3, 3), P: (1, 3, N_points)
    Xc = torch.bmm(R, P.expand(NUM_VIEWS, -1, -1))  # (N_views, 3, N_points)
    Xc = Xc + translations.unsqueeze(2)  # (N_views, 3, N_points) + (N_views, 3, 1)

    Xc_x = Xc[:, 0, :]  # (N_views, N_points)
    Xc_y = Xc[:, 1, :]
    Xc_z = Xc[:, 2, :]

    # Projection: u = -f * Xc/Zc + cx,  v = f * Yc/Zc + cy
    # Note: negative sign on X to prevent left-right flip when Zc < 0
    u = -focal * Xc_x / Xc_z + CX
    v = focal * Xc_y / Xc_z + CY

    projected = torch.stack([u, v], dim=-1)  # (N_views, N_points, 2)
    return projected


# ============================================================
# Bundle Adjustment Optimization
# ============================================================
def run_bundle_adjustment():
    """Main optimization loop."""

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
        print(f"Using GPU: {GPU_ID}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data
    points2d_npz, colors = load_data()
    obs_2d, vis_mask = prepare_observation_tensors(points2d_npz, device)

    print(f"Data loaded: {NUM_VIEWS} views, {NUM_POINTS} points")
    print(f"Visible points per view: {vis_mask.sum(dim=1).tolist()[:5]}...")

    # ---- Initialize optimizable parameters ----

    # Focal length: f = H / (2 * tan(fov/2))
    init_focal = IMAGE_HEIGHT / (2.0 * np.tan(np.radians(INIT_FOV_DEG / 2.0)))
    focal = torch.nn.Parameter(torch.tensor(init_focal, dtype=torch.float32, device=device))
    print(f"Initial focal length: {init_focal:.1f}")

    # Euler angles for each camera (initialized to zero = identity rotation)
    euler_angles = torch.nn.Parameter(
        torch.zeros(NUM_VIEWS, 3, dtype=torch.float32, device=device)
    )

    # Translation for each camera: [0, 0, -d]
    translations = torch.nn.Parameter(
        torch.tensor(
            [[0.0, 0.0, -INIT_DISTANCE]] * NUM_VIEWS,
            dtype=torch.float32,
            device=device,
        )
    )

    # 3D points: random initialization near origin
    torch.manual_seed(42)
    points3d = torch.nn.Parameter(
        torch.randn(NUM_POINTS, 3, dtype=torch.float32, device=device) * 0.3
    )

    # ---- Optimizer ----
    # Use separate parameter groups with different learning rates
    optimizer = torch.optim.Adam([
        {"params": [points3d], "lr": LR_POINTS},
        {"params": [euler_angles, translations], "lr": LR_CAMERA},
        {"params": [focal], "lr": LR_FOCAL},
    ])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    # ---- Training Loop ----
    loss_history = []

    print(f"\nStarting optimization for {NUM_ITERATIONS} iterations...")
    for iteration in range(NUM_ITERATIONS):
        optimizer.zero_grad()

        # Project 3D points to 2D
        pred_2d = project_points(points3d, euler_angles, translations, focal)

        # Compute reprojection error (L2 distance)
        error = pred_2d - obs_2d  # (N_views, N_points, 2)
        error_per_point = (error ** 2).sum(dim=-1)  # (N_views, N_points)

        # Mask by visibility and compute mean loss
        masked_error = error_per_point * vis_mask  # zero out invisible points
        loss = masked_error.sum() / vis_mask.sum()

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if iteration % 100 == 0 or iteration == NUM_ITERATIONS - 1:
            # Compute RMSE in pixels for visible points
            rmse = np.sqrt(loss_val)
            print(f"  Iter {iteration:4d}/{NUM_ITERATIONS} | "
                  f"Loss: {loss_val:.4f} | RMSE: {rmse:.2f} px | "
                  f"f: {focal.item():.1f}")

    print("Optimization complete!")

    # ---- Save Results ----

    # 1. Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Reprojection Loss (MSE)")
    plt.title("Bundle Adjustment Optimization")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to {OUTPUT_DIR}/loss_curve.png")

    # 2. Reconstructed 3D point cloud as OBJ
    points3d_np = points3d.detach().cpu().numpy()
    colors_np = colors  # (20000, 3), already in [0, 1]

    obj_path = os.path.join(OUTPUT_DIR, "reconstructed.obj")
    with open(obj_path, "w") as f:
        for i in range(NUM_POINTS):
            x, y, z = points3d_np[i]
            r, g, b = colors_np[i]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
    print(f"Point cloud saved to {obj_path}")

    # 3. Save final parameters
    params_path = os.path.join(OUTPUT_DIR, "params.npz")
    np.savez(
        params_path,
        points3d=points3d_np,
        euler_angles=euler_angles.detach().cpu().numpy(),
        translations=translations.detach().cpu().numpy(),
        focal=np.array([focal.item()]),
        loss_history=np.array(loss_history),
    )
    print(f"Parameters saved to {params_path}")

    # 4. Final statistics
    final_rmse = np.sqrt(loss_history[-1])
    print(f"\n=== Final Results ===")
    print(f"  Final RMSE: {final_rmse:.2f} pixels")
    print(f"  Optimized focal length: {focal.item():.1f}")
    print(f"  Point cloud center: {points3d_np.mean(axis=0)}")
    print(f"  Point cloud std: {points3d_np.std(axis=0)}")

    return loss_history


if __name__ == "__main__":
    run_bundle_adjustment()
