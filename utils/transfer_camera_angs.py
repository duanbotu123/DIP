import cv2
import numpy as np
import json
# 将相机格式转化为angs的，可能会用到

def read_extri(extri_path):
    fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    names_node = fs.getNode("names")
    names = []
    for i in range(names_node.size()):
        names.append(names_node.at(i).string())
    cameras = {}
    for name in names:
        R = fs.getNode(f"Rot_{name}").mat().flatten().tolist()
        T = fs.getNode(f"T_{name}").mat().flatten().tolist()
        cameras[name] = {'R': R, 'T': T}
    
    return cameras

def read_intri(intri_path):
    fs = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    names_node = fs.getNode("names")
    names = []
    for i in range(names_node.size()):
        names.append(names_node.at(i).string())
    cameras = {}
    for name in names:
        K = fs.getNode(f"K_{name}").mat().flatten().tolist()
        dist = fs.getNode(f"dist_{name}").mat().flatten().tolist()
        cameras[name] = {'K': K, 'distCoeff': dist}
    
    return cameras

intri_path = '/data1/hlp/dataset/241129/human/intri.yml'
extri_path = '/data1/hlp/dataset/241129/human/extri.yml'
save_path = '/nas_data/home/hlp/data/angs/zlb/calibration.json'
fourK = [3840, 2160]
oneK = [1920, 1080]

names = ['3','5','9', '10', '11', '13', '14']

cameras = {}
intri = read_intri(intri_path)
extri = read_extri(extri_path)
i = 0
for name in names:
    camera = {}
    K = intri[name]['K']
    dist = intri[name]['distCoeff']
    R = extri[name]['R']
    T = extri[name]['T']
    camera['K'] = K
    camera['R'] = R
    camera['T'] = T
    camera['distCoeff'] = dist
    camera['imgSize'] = fourK
    camera['rectifyAlpha'] = 0.0
    angs_name = f'cam{i:02d}'
    cameras[angs_name] = camera
    i += 1
with open(save_path, 'w') as json_file:
    json.dump(cameras, json_file, indent=4)