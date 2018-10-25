import math

import torch
from torch.nn import Module, ModuleList
from group_capsules.transforms import pseudo_cartesian
from group_capsules.transforms.coordinate import posevec_to_rotmat
from torch_geometric.nn import SplineConv


class GroupConvLayer(Module):
    def __init__(self,
                 in_channels,
                 num_poses,
                 kernel_size,
                 channels_per_pose=1,
                 bias=True,
                 initial=False,
                 conv_type='SplineConv'):
        super(GroupConvLayer, self).__init__()

        self.in_channels = in_channels
        self.num_poses = num_poses
        self.initial = initial
        self.conv_type = conv_type

        if initial:
            self.convs = ModuleList([
                SplineConv(in_channels, num_poses*channels_per_pose, 2,
                           kernel_size, bias=bias, norm=False,
                           root_weight=False)
                ])
        else:
            self.convs = ModuleList([
                SplineConv(in_channels, channels_per_pose, 2, kernel_size,
                           bias=bias, norm=False, root_weight=False)
                for _ in range(num_poses)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, pos, pose):

        pseudo = pseudo_cartesian(edge_index, pos, size=math.sqrt(2))
        row, col = edge_index
        out = []

        if self.initial:
            pose = pose.detach().view(-1,1,2)
            pose_mats = posevec_to_rotmat(pose)
            pose_mat = pose_mats[:, 0][row]  # e
            p = torch.matmul(pose_mat, pseudo.unsqueeze(-1) - 0.5) + 0.5
            p = p.squeeze(-1)

            f_out = self.convs[0](x, edge_index, p).squeeze(-1)
        else:
            pose = pose.detach().view(-1,self.num_poses,2)
            pose_mats = posevec_to_rotmat(pose)
            for i in range(self.num_poses):
                pose_mat = pose_mats[:, i][row]  # e
                p = torch.matmul(pose_mat, pseudo.unsqueeze(-1) - 0.5) + 0.5
                p = p.squeeze(-1)
                out.append(self.convs[i](x, edge_index, p).squeeze(-1))
            f_out = torch.stack(out, dim=-1)

        return f_out

    def __repr__(self):
        return '{}({}, {})'.format(
            self.__class__.__name__, self.conv.in_channels,
            self.conv.out_channels)
