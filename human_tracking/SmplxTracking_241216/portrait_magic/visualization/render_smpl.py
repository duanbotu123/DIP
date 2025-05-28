import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np
from tqdm import tqdm
import torch
from pytorch3d.io import load_obj
from ..dmm_models import FLAME, flame_config, SMPLX, smplx_model_path
from ..visualization.visualizer import Renderer, merge, load_image, images_to_video
import argparse
import cv2


def load_cameras(camera_path: str):
    """Read intri/extri yaml files and return camera dict compatible with *vis_smpl*."""
    extri_file = os.path.join(camera_path, 'extri.yml')
    intri_file = os.path.join(camera_path, 'intri.yml')
    if not (os.path.isfile(extri_file) and os.path.isfile(intri_file)):
        raise FileNotFoundError('extri.yml / intri.yml not found in ' + camera_path)
    fs_extri = cv2.FileStorage(extri_file, cv2.FILE_STORAGE_READ)
    fs_intri = cv2.FileStorage(intri_file, cv2.FILE_STORAGE_READ)
    names_node = fs_extri.getNode('names')
    names = [names_node.at(i).string() for i in range(names_node.size())]
    cams = {}
    for n in names:
        cams[n] = {
            'R': fs_extri.getNode(f'Rot_{n}').mat().astype(np.float32),
            'T': fs_extri.getNode(f'T_{n}').mat().astype(np.float32),
            'K': fs_intri.getNode(f'K_{n}').mat().astype(np.float32),
        }
    fs_extri.release()
    fs_intri.release()
    return cams

def vis_smpl(cameras, vertices, faces, images, save_folder, nf, sub_vis, add_back=True):
    os.makedirs(os.path.join(save_folder, 'vposer_smplx'), exist_ok=True)
    outname = os.path.join(save_folder, 'vposer_smplx', '{:06d}.jpg'.format(nf))
    render_data = {}
    assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
    pid = 0
    render_data[pid] = {'vertices': vertices, 'faces': faces, 
        'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
    sub_cameras = {'K': [], 'R':[], 'T':[]}
    for key in sub_cameras.keys():
        sub_cameras[key] = np.stack([cameras[cam][key] for cam in sub_vis])
    images = images
    _vis_smpl(render_data, images, sub_cameras, outname, add_back=add_back)

def _vis_smpl(render_data, images, cameras, outname, add_back):
    render = Renderer(height=1024, width=1024, faces=None)
    render_results = render.render(render_data, cameras, images, add_back=add_back)
    image_vis = merge(render_results)
    cv2.imwrite(outname, image_vis)
    return image_vis

def main():
    parser = argparse.ArgumentParser(
        description="Render SMPL‑X meshes over multi‑view images and save per‑frame jpg + video"
    )
    parser.add_argument("--data", type=str, required=True,help="Path to the data folder")
    parser.add_argument("--sub_vis", nargs="+", default=[], help="List of camera ids to visualize")
    parser.add_argument("--stride", type=int, default=2, help="Render every n‑th frame (default 2, only even frames)")
    parser.add_argument("--shape_dim", type=int, default=10)
    parser.add_argument("--expr_dim", type=int, default=10)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    img_folder = os.path.join(args.data, "images")
    pose_file = os.path.join(args.data, "thuman4", "vposer_smpl_params.npz")
    camera_path = args.data
    cameras = load_cameras(camera_path)
    save_folder = os.path.join(args.data, "body_track")
    render_folder = os.path.join(save_folder, "vposer_smplx")
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(render_folder, exist_ok=True)
    # SMPL‑X model
    smplx_model = SMPLX(
        smplx_model_path,
        num_expression_coeffs=args.expr_dim,
        num_betas=args.shape_dim,
        use_face_contour=True,
        use_pca=False,
    ).to(device).eval()

    _, faces_idx, _ = load_obj(os.path.join(smplx_model_path, "smplx_uv.obj"))
    faces = faces_idx.verts_idx.cpu().numpy()

    # Load parameters
    params = np.load(pose_file)

    frames = params["body_pose"].shape[0]


    for nf in tqdm(range(frames), desc="render"):
        if nf % args.stride != 0:
            continue

        images = load_image(img_folder, args.sub_vis, nf)  # user provided helper

        smplx_out = smplx_model(
            betas=torch.from_numpy(params["betas"]).to(device),
            body_pose=torch.from_numpy(params["body_pose"][nf : nf + 1]).to(device),
            left_hand_pose=torch.from_numpy(params["left_hand_pose"][nf : nf + 1]).to(device),
            right_hand_pose=torch.from_numpy(params["right_hand_pose"][nf : nf + 1]).to(device),
            jaw_pose=torch.from_numpy(params["jaw_pose"][nf : nf + 1]).to(device),
            expression=torch.from_numpy(params["expression"][nf : nf + 1]).to(device),
            global_orient=torch.from_numpy(params["global_orient"][nf : nf + 1]).to(device),
            transl=torch.from_numpy(params["transl"][nf : nf + 1]).to(device),
            leye_pose=torch.from_numpy(params["leye_pose"][nf : nf + 1]).to(device),
            reye_pose=torch.from_numpy(params["reye_pose"][nf : nf + 1]).to(device),
            with_iris_return=False,
        )

        verts = smplx_out.vertices.squeeze(0).cpu().numpy()
        vis_smpl(cameras, verts, faces, images, save_folder, nf, args.sub_vis, add_back=True)

    # Convert to video
    video_path = os.path.join(save_folder, "vposer_smplx.mp4")
    images_to_video(render_folder, video_path)
    print("Saved video to", video_path)


if __name__ == "__main__":
    main()


