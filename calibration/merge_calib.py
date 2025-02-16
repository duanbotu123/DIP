from math import e
import cv2
import numpy as np
import json
import argparse

from tqdm import trange

def read_extri(extri_path):
    '''
    Read extrinsic parameters from a yml file
    Args:
        extri_path: str, path to extrinsic parameters file
    Returns:
        cameras: dict, extrinsic parameters of all cameras
    '''
    fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    names_node = fs.getNode("names")
    names = [names_node.at(i).string() for i in range(names_node.size())]
    cameras = {name: {
        'R': fs.getNode(f"Rot_{name}").mat().flatten().tolist(),
        'T': fs.getNode(f"T_{name}").mat().flatten().tolist()
    } for name in names}
    fs.release()
    return cameras

def cal_trans(extri1, extri2):
    '''
    Calculate the transformation matrix between two cameras
    Args:
        extri1: dict, extrinsic parameters of camera 1
        extri2: dict, extrinsic parameters of camera 2
    Returns:
        R: np.array, rotation matrix
        T: np.array, translation vector
    '''
    rotation = []
    translation = []
    for camera in extri1.keys():
        if camera in extri2.keys():
            Rc1 = np.array(extri1[camera]['R']).reshape(3, 3)
            Rc2 = np.array(extri2[camera]['R']).reshape(3, 3)
            R1 = Rc1.T
            R2 = Rc2.T
            T1 = - np.dot(Rc1.T, np.array(extri1[camera]['T']))
            T2 = - np.dot(Rc2.T, np.array(extri2[camera]['T']))
            # T1 = np.array(extri1[camera]['T'])
            # T2 = np.array(extri2[camera]['T'])
            delta_t = T1 - np.dot(R1,np.dot(R2.T,T2))
            delta_R = R1 @ R2.T
            rotation.append(delta_R)
            translation.append(delta_t)
    delta_R = np.mean(rotation, axis=0)
    delta_t = np.mean(translation, axis=0)
    return delta_R, delta_t

def apply_trans(dR, dT, extri1, extri2):
    '''
    Apply transformation to extrinsic parameters and merge two cameras system
    Args:
        dR: np.array, rotation matrix
        dT: np.array, translation vector
        extri1: dict, extrinsic parameters
        extri2: dict, extrinsic parameters
    Returns:
        extri: dict, transformed extrinsic parameters
    '''
    extri = {}
    for camera in extri1.keys():
        extri[camera] = {
            'R': extri1[camera]['R'],
            'T': extri1[camera]['T']
        }
    for camera in extri2.keys():
        if camera in extri1.keys():
            Rc1 = np.array(extri1[camera]['R']).reshape(3, 3)
            Rc2 = np.array(extri2[camera]['R']).reshape(3, 3)
            R2 = Rc2.T
            T2 = np.array(extri2[camera]['T'])
            T2 = - np.dot(Rc2.T, T2)
            rot = np.dot(dR, R2)
            transl = np.dot(dR, T2) + dT
            rot1 = Rc1.T
            transl1 = - np.array(extri1[camera]['T']) @ Rc1
            rot2 = Rc2.T
            transl2 = - np.array(extri2[camera]['T']) @ Rc2
            print(f'{camera}:')
            print(f'R:{rot}')
            print(f'T:{transl}')
            print(f'R1:{rot1}')
            print(f'T1:{transl1}')
            print(f'R2:{rot2}')
            print(f'T2:{transl2}')
            print(f'delta_t:{transl1 - transl2}')
            print(f'dt:{transl - transl1}')
            print('\n')
            continue
        Rc2 = np.array(extri2[camera]['R']).reshape(3, 3)
        R2 = Rc2.T
        T2 = np.array(extri2[camera]['T'])
        T2 = - np.dot(Rc2.T, T2)
        rot = np.dot(dR, R2)
        print(f'{camera}:')
        transl = np.dot(dR, T2) + dT
        print(f'temp:{transl}')
        transl = - np.dot(rot.T, transl)
        print(f'R:{rot}')
        print(f't:{transl}')
        
        extri[camera] = {
            'R': rot.T.flatten().tolist(),
            'T': transl.tolist()
        }
    return extri

def write_extri(extri, output_path):
    '''
    Write extrinsic parameters to a yml file
    Args:
        extri: dict, extrinsic parameters
        output_path: str, path to output yml file
    '''
    fs = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
    
    # Write names as a list in the specified format
    names = list(extri.keys())
    fs.startWriteStruct("names", cv2.FileNode_SEQ)
    for name in names:
        fs.write("", name)  # Write each name as a sequence element
    fs.endWriteStruct()
    
    # Write rotation and translation parameters for each camera
    for camera, params in extri.items():
        rvec, _ = cv2.Rodrigues(np.array(params['R']).reshape(3, 3))
        fs.write(f"Rot_{camera}", np.array(params['R']).reshape(3, 3))
        fs.write(f"R_{camera}", rvec)
        # fs.write(f"T_{camera}", (- np.array(params['T'] @ np.array(params['R']).reshape(3, 3))))
        fs.write(f"T_{camera}", np.array(params['T']))
    
    fs.release()

def main():
    parser = argparse.ArgumentParser(description="Merge camera parameters to an uniform world coordinate system")
    parser.add_argument("--ex1", type=str, help="Path to extri.yml")
    parser.add_argument("--ex2", type=str, help="Path to extri.yml")
    parser.add_argument("--output", type=str, help="Path to output merged extri.yml")
    args = parser.parse_args()
    extri1 = read_extri(args.ex1)
    extri2 = read_extri(args.ex2)
    dr, dt = cal_trans(extri1, extri2)
    extri = apply_trans(dr, dt, extri1, extri2)
    write_extri(extri, args.output)

if __name__ == "__main__":
    main()