import torch

def a_in_b_torch(a, b):
    ainb = torch.isin(a, b)
    return ainb  