import torch
import numpy as np

def estimate_transform_2d(src_pts, dst_pts, mode='similarity'):
    # src_pts: (b, n, 2), dst_pts: (b, n, 2)
    # return: (b, 2, 3)
    if mode == 'similarity':
        b, n = src_pts.shape[:2]
        x, y = src_pts[:, :, :1], src_pts[:, :, 1:]
        x_d, y_d = dst_pts[:, :, :1], dst_pts[:, :, 1:]
        vec_1, vec_0 = torch.ones_like(x), torch.zeros_like(x)
        A_1 = torch.cat((x, y, vec_1, vec_0), dim=-1)
        A_2 = torch.cat((y, -x, vec_0, vec_1), dim=-1)
        A_mat = torch.cat((A_1, A_2), dim=-1).reshape(b, 2*n, 4)
        b_vec = torch.cat((x_d, y_d), dim=-1).reshape(b, 2*n, 1)
        x_vec = torch.linalg.lstsq(A_mat, b_vec)[0]   # (b, 4, 1)
        return torch.cat((x_vec[:, 0], x_vec[:, 1], x_vec[:, 2], -x_vec[:, 1], x_vec[:, 0], x_vec[:, 3]), dim=-1).reshape(b, 2, 3)
    elif mode == 'crop':
        b, n = src_pts.shape[:2]
        x, y = src_pts[:, :, :1], src_pts[:, :, 1:]
        x_d, y_d = dst_pts[:, :, :1], dst_pts[:, :, 1:]
        vec_1, vec_0 = torch.ones_like(x), torch.zeros_like(x)
        A_1 = torch.cat((x, vec_1, vec_0), dim=-1)
        A_2 = torch.cat((y, vec_0, vec_1), dim=-1)
        A_mat = torch.cat((A_1, A_2), dim=-1).reshape(b, n*2, 3)
        b_vec = torch.cat((x_d, y_d), dim=-1).reshape(b, n*2, 1)
        x_vec = torch.linalg.lstsq(A_mat, b_vec)[0]   # (b, 3, 1)
        zero_vec = torch.zeros_like(x_vec[:, 0])
        return torch.cat((x_vec[:, 0], zero_vec, x_vec[:, 1], zero_vec, x_vec[:, 0], x_vec[:, 2]), dim=-1).reshape(b, 2, 3)
    else:
        raise NotImplementedError

def extract_5p(lms, lms_mode='300w'):
    if lms_mode == '300w':
        lm_leye = torch.mean(lms[:, [36, 39]], dim=1, keepdim=True)
        lm_reye = torch.mean(lms[:, [42, 45]], dim=1, keepdim=True)
        lm_nose = lms[:, 30:31]
        lm_lmouth = lms[:, 48:49]
        lm_rmouth = lms[:, 54:55]
        return torch.cat((lm_leye, lm_reye, lm_nose, lm_lmouth, lm_rmouth), dim=1)
    elif lms_mode == 'wflw':
        lm_leye = torch.mean(lms[:, [60, 64]], dim=1, keepdim=True)
        lm_reye = torch.mean(lms[:, [68, 72]], dim=1, keepdim=True)
        lm_nose = lms[:, 54:55]
        lm_lmouth = lms[:, 76:77]
        lm_rmouth = lms[:, 82:83]
        return torch.cat((lm_leye, lm_reye, lm_nose, lm_lmouth, lm_rmouth), dim=1)
    elif lms_mode == 'mediapipe':
        lm_idx = np.array([4, 33, 133, 398, 263, 61, 291], dtype=np.int64)
        lm5p = torch.cat((lms[:, lm_idx[0]].unsqueeze(1), torch.mean(lms[:, lm_idx[[1, 2]]], dim=1).unsqueeze(1), torch.mean(lms[:, lm_idx[[3, 4]]], dim=1).unsqueeze(1), lms[:, lm_idx[5]].unsqueeze(1), lms[:, lm_idx[6]].unsqueeze(1)), dim=1)
        return lm5p[:, [1, 2, 0, 3, 4], :]
    else:
        raise NotImplementedError

def estimate_transform_id(pts, lms_mode='300w', crop_size=112):
    lms_5p = extract_5p(pts, lms_mode)
    dst_5p = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [
                      41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    dst_5p = dst_5p/112.*crop_size
    dst_5p = torch.as_tensor(dst_5p, device=lms_5p.device).unsqueeze(
        0).expand(lms_5p.shape[0], -1, -1)
    return estimate_transform_2d(lms_5p, dst_5p)
