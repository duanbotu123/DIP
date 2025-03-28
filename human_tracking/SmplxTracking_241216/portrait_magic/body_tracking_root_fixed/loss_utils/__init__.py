import torch

def cal_euclidean_distance(x1, x2, confidence = None):
    """
    x1: (b, n, d), x2: (b, n, d), optional (b, n, 1) -> (b)
    """
    if confidence is None:
        return torch.mean(torch.mean(torch.sqrt((torch.sum((x1-x2)**2, dim=-1) + 1e-7)), dim=-1))
    else:
        last_dim_dis = torch.sqrt((torch.sum((x1-x2)**2, dim=-1) + 1e-7))
        confidence = confidence.squeeze(-1) + 1e-6
        last_dim_dis = torch.mean(last_dim_dis*confidence, dim=-1)/torch.mean(confidence, dim=-1)
        return torch.mean(last_dim_dis)

def cal_l2_loss(x, valid_mask = None):
    if valid_mask is None:
        return torch.mean(x**2)
    else:
        return torch.mean(x**2 * valid_mask) / (torch.mean(valid_mask) + 1e-6)
    

def cal_similarity_loss(x1, x2, cos, valid_mask = None):
   cos_sim = cos(x1, x2)
   return torch.mean((1. - cos_sim)*valid_mask) / (torch.mean(valid_mask + 1e-6))
    
    
def cal_smooth_loss(x, with_grad = False, attn_weight = None):
    '''
    x: (n, d)
    '''
    if attn_weight is None:
        attn_weight = torch.ones_like(x[:1])
    if not with_grad:
        return torch.mean(((x[1:-1] - (x[2:] + x[:-2])*.5)**2) * attn_weight) / (torch.mean(attn_weight) + 1e-5)
    return torch.mean((torch.abs(x[:-1] - x[1:])) * attn_weight) / (torch.mean(attn_weight) + 1e-5)