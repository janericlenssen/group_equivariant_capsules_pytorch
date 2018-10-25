import torch


def cartesian_to_polar(cartesian, is_unit=False):
    theta = torch.atan2(cartesian[..., 0], cartesian[..., 1])
    polar = (theta.unsqueeze(-1) if is_unit else torch.stack(
        [cartesian.norm(p=2, dim=-1), theta], dim=-1))
    return polar


def polar_to_cartesian(polar):
    theta = polar[..., -1]
    cartesian = torch.stack([theta.cos(), theta.sin()], dim=-1)
    cartesian = cartesian if polar.size(-1) == 1 else cartesian * polar[..., 0]
    return cartesian


def posevec_to_rotmat(posevec):
    x = posevec[...,0]
    y = posevec[...,1]
    mat = torch.stack([x, -y, y, x], dim=-1)
    return mat.view(*mat.size()[:-1], 2, 2)

def inverse_pose(posevec):
    return torch.stack([posevec[...,0], -posevec[...,1]], dim=-1)