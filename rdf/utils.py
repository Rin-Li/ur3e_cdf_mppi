import numpy as np
import torch
import os
import trimesh
import glob
CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def transform_points(points, trans, device):
    """Transform points in SE(3).  points: (N,3)  trans: (B,4,4)"""
    B, N = trans.shape[0], points.shape[0]
    ones = torch.ones([B, N, 1], device=device).float()
    points_ = torch.cat([points.unsqueeze(0).expand(B, N, 3), ones], dim=-1)
    points_ = torch.matmul(trans, points_.permute(0, 2, 1)).permute(0, 2, 1)
    return points_[:, :, :3].float()


def mse(yhat, y):
    return torch.nn.MSELoss(reduction='mean')(yhat, y)


def rmse(yhat, y):
    return torch.sqrt(mse(yhat, y))


def print_eval(yhat, y, string='default'):
    yhat, y = yhat.view(-1), y.view(-1)
    y_near = (y.abs() < 0.03)
    y_far  = (y.abs() > 0.03)
    MAE  = (yhat - y).abs().mean()
    MSE  = mse(yhat, y)
    RMSE = rmse(yhat, y)
    MAE_near  = (yhat[y_near] - y[y_near]).abs().mean()
    MSE_near  = mse(yhat[y_near], y[y_near])
    RMSE_near = rmse(yhat[y_near], y[y_near])
    MAE_far   = (yhat[y_far] - y[y_far]).abs().mean()
    MSE_far   = mse(yhat[y_far], y[y_far])
    RMSE_far  = rmse(yhat[y_far], y[y_far])
    res = [MAE, MSE, RMSE, MAE_near, MSE_near, RMSE_near, MAE_far, MSE_far, RMSE_far]
    return [r.item() for r in res]
