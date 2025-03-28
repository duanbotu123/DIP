from rembg import new_session, remove
import cv2
from tqdm import tqdm


import onnxruntime as ort
print(ort.get_device())

img = cv2.imread('data/test_dir/ori_imgs/000000.jpg')

for i in tqdm(range(50)):
    output = remove(img)
    