import json
import os
import cv2
import numpy as np

raw_folder = '/data1/hlp/dataset/241129/human/output/smplx/smpl_full_origin'
joint_folder = '/data1/hlp/dataset/241129/human/output/smplx/joints'
out_folder = '/data1/hlp/dataset/241129/human/output/smplx/smpl_full'
os.makedirs(out_folder, exist_ok=True)
for json_file in sorted(os.listdir(raw_folder)):
    rot_file_path = os.path.join(raw_folder, json_file)
    joint_file_path = os.path.join(joint_folder, json_file) 
    with open(rot_file_path, 'r') as file:
        data = json.load(file)
        data_copy = data
        data0 = data[0]
        Rh = np.array(data0["Rh"][0])
        Th = np.array(data0["Th"][0])
        rotation, _ = cv2.Rodrigues(Rh)
    with open(joint_file_path, 'r') as file:
        data = json.load(file)[0]
        joints = np.array(data["joints"])[0]
        j0 = joints[0,:]
    smpl_j = np.matmul(rotation.T,(j0 - Th))
    print(smpl_j)
    Tnew = Th - smpl_j + rotation@smpl_j
    print(Tnew)
    data_copy[0]["Th"] = [Tnew.tolist()]
    json_save = os.path.join(out_folder, json_file)
    with open(json_save, 'w') as file:
        json.dump(data_copy, file, indent=4)
        print(f'data has been writen into {json_save}')