import cv2
import numpy as np
import json
import argparse

def read_extri(extri_path):
    fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    names_node = fs.getNode("names")
    names = [names_node.at(i).string() for i in range(names_node.size())]
    cameras = {}
    for name in names:
        R = fs.getNode(f"Rot_{name}").mat().flatten().tolist()
        T = fs.getNode(f"T_{name}").mat().flatten().tolist()
        cameras[name] = {'R': R, 'T': T}
    return cameras

def read_intri(intri_path):
    fs = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    names_node = fs.getNode("names")
    names = [names_node.at(i).string() for i in range(names_node.size())]
    cameras = {}
    for name in names:
        K = fs.getNode(f"K_{name}").mat().flatten().tolist()
        dist = fs.getNode(f"dist_{name}").mat().flatten().tolist()
        cameras[name] = {'K': K, 'distCoeff': dist}
    return cameras, names

def main(args):
    intri, intri_names = read_intri(args.intri)
    extri = read_extri(args.extri)

    # 如果没有指定 --names，则用 intri 中读取的名称列表
    if args.names:
        camera_names = args.names.split(",")
    else:
        camera_names = intri_names

    img_size = [int(v) for v in args.img_size.split("x")]
    
    cameras = {}
    for i, name in enumerate(camera_names):
        if name not in intri or name not in extri:
            print(f"Warning: Camera {name} not found in both intri and extri.")
            continue

        camera = {
            'K': intri[name]['K'],
            'R': extri[name]['R'],
            'T': extri[name]['T'],
            'distCoeff': intri[name]['distCoeff'],
            'imgSize': img_size,
            'rectifyAlpha': 0.0
        }
        cam_name = f'cam{i:02d}'
        cameras[cam_name] = camera

    with open(args.output, 'w') as json_file:
        json.dump(cameras, json_file, indent=4)
    print(f"Calibration saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert intri/extri YAML to angs-style JSON calibration file.")
    parser.add_argument('--intri', type=str, required=True, help="Path to intri.yml")
    parser.add_argument('--extri', type=str, required=True, help="Path to extri.yml")
    parser.add_argument('--output', type=str, required=True, help="Path to save calibration JSON")
    parser.add_argument('--names', type=str, default="", help='Comma-separated camera names (e.g., "3,5,9,10"). If not set, use all in intri.')
    parser.add_argument('--img_size', type=str, default="1920x1080", help='Image size in WIDTHxHEIGHT (default: 3840x2160)')

    args = parser.parse_args()
    main(args)
