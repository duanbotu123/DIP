import numpy as np

txt_path = '/nas_data/home/hlp/data/smpl_recon_test/ldmks/000000.wb'

ldmks = np.loadtxt(txt_path, dtype=np.float32)
print(ldmks.shape)