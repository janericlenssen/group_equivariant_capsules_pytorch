from math import sin, cos

import torch


def rotate_cartesian(pseudo, theta):
    s, c = sin(theta), cos(theta)
    pseudo = torch.matmul(pseudo, pseudo.new([[c, -s], [s, c]]))
    return pseudo


def rotate_by_pose(pseudo, pos):
    pass
