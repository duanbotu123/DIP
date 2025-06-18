import torch

def cal_euclidean_distance(x1, x2, confidence = None, weights = None):
    """
    x1: (b, n, d), x2: (b, n, d), optional (b, n, 1), optional (b) -> (b)
    """
    b = x1.shape[0]
    if weights is None:
        weights = torch.ones(b, device=x1.device)
    if confidence is None:
        return torch.mean(torch.mean(torch.sqrt((torch.sum((x1-x2)**2, dim=-1) + 1e-7)), dim=-1))
    else:
        last_dim_dis = torch.sqrt((torch.sum((x1-x2)**2, dim=-1) + 1e-7))
        confidence = confidence.squeeze(-1) + 1e-6
        last_dim_dis = torch.mean(last_dim_dis*confidence, dim=-1)/torch.mean(confidence, dim=-1)
        # print(f'last dim dis shape:{last_dim_dis.shape}')
        # print(f'last dim value:{last_dim_dis[0]}')
        # return torch.mean(last_dim_dis)
        return torch.sum(last_dim_dis * weights) / torch.sum(weights)
    
def cal_euclidean_distance_mv(x1, x2, confidence=None, weights=None):
    '''
    x1: (v, b, n, d)
    x2: (v, b, n, d)
    confidence: optional, (v, b, n, 1)
    weights: optional, (v, b)
    
    Returns:
        (b) tensor, 每个 batch 的加权距离, weights 对所有 view 重新归一化
    '''
    v, b, n, d = x1.shape
    device = x1.device

    # 若未指定 weights，则默认每个视角权重为 1
    if weights is None:
        weights = torch.ones(v, b, device=device)

    if confidence is None:
        # 计算欧氏距离，形状：(v, b, n)
        dist = torch.sqrt(torch.sum((x1 - x2)**2, dim=-1) + 1e-7)
        # 对每个视角和 batch，在 n 维度上取平均，得到 (v, b)
        dist_mean = torch.mean(dist, dim=-1)
    else:
        # 计算欧氏距离，形状：(v, b, n)
        dist = torch.sqrt(torch.sum((x1 - x2)**2, dim=-1) + 1e-7)
        # 调整 confidence 的形状：(v, b, n)
        conf = confidence.squeeze(-1) + 1e-6
        # 对每个视角和 batch，计算 confidence 加权的距离平均，形状：(v, b)
        dist_mean = torch.mean(dist * conf, dim=-1) / torch.mean(conf, dim=-1)

    # 对每个 batch，将各个视角的距离进行加权平均
    # weights 的形状为 (v, b)，对每个 batch，需要先对 view 方向归一化权重
    # 这里计算： result[b] = sum_v(dist_mean[v,b]*weights[v,b]) / sum_v(weights[v,b])
    weighted_sum = torch.sum(dist_mean * weights, dim=0)  # (b)
    weights_sum = torch.sum(weights, dim=0)                # (b)
    result = weighted_sum / weights_sum                   # (b)
    return result


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