import torch
import smplx
import trimesh
import numpy as np

'''This file constains most kinds of losses needed.'''

class Losses(object):
    def __init__(self):
        super.__init__()

    '''velocity loss of human'''
    @staticmethod
    def velocity_loss_h(smplx):
        v_l = 0.
        for key, val in smplx.items():
            if key != 'betas':# and key in ['left_hand_pose', 'right_hand_pose']:
                velocity = torch.abs(val[1:, ...] - val[:-1, ...])
                # print('vs', velocity.shape)
                v_l += torch.mean(velocity)
        return v_l
    
    '''velocity loss of objects'''
    @staticmethod
    def velocity_loss_o(trans):
        v_l = 0.
        velocity = torch.abs(trans[1:, ...] - trans[:-1, ...])
        # print('vs', velocity.shape)
        v_l += torch.mean(velocity)
        return v_l
    
    '''contact loss of objects'''
    @staticmethod
    def contact_loss(smplx, object):
        v_l = 0.
        for key, val in smplx.items():
            if key != 'betas':# and key in ['left_hand_pose', 'right_hand_pose']:
                velocity = torch.abs(val[1:, ...] - val[:-1, ...])
                # print('vs', velocity.shape)
                v_l += torch.mean(velocity)
        return v_l
    @staticmethod
    def sdf_loss(o_mesh, h_sdf):
        #$ sample points on the object mesh
        points = 0
        #$ add [R, T] on these points
        #$ calculate the sdfs of these points
        #$ calculate the loss
        return sdf_loss


