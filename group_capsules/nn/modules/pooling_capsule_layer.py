import torch
from torch.nn import Module, Parameter, Linear
import torch.nn.functional as F
from group_capsules.transforms.coordinate import posevec_to_rotmat, inverse_pose
from group_capsules.utils.grid import pool_grid

# currently only works if poolingsize*k = inputsize for some natural k
def space_to_depth(input, block_size):
    block_size = int(block_size)
    block_size_sq = block_size * block_size
    output = input
    (batch_size, s_height, s_width, s_depth, s_posev) = output.size()
    d_depth = s_depth * block_size_sq
    d_height = int((s_height + (block_size - 1)) / block_size)
    t_1 = output.split(block_size, 2)
    stack = [
        t_t.contiguous().view(batch_size, d_height, 1, d_depth, s_posev)
        for t_t in t_1
    ]
    output = torch.cat(stack, 2)
    return output



class PoolingCapsuleLayer(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_iterations=3,
                 pool_size=2,
                 f_in_size=None):
        super(PoolingCapsuleLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if f_in_size is not None:
            self.f_in_size = f_in_size
        else:
            self.f_in_size = self.in_channels
        self.num_iterations = num_iterations
        self.pool_length = pool_size
        self.pool_size = int(pool_size**2)

        self.alpha = Parameter(torch.Tensor(1))
        self.beta = Parameter(torch.Tensor(1))

        self.lin = Linear(self.f_in_size, out_channels)

        self.reset_parameters()
        self.pool_positions = pool_grid(self.pool_size)

        self.theta_gen1 = Linear(2 * self.in_channels, 32, bias=False)
        self.theta_gen2 = Linear(32, out_channels * in_channels, bias=False)

    def reset_parameters(self):
        self.alpha.data.fill_(1)
        self.beta.data.fill_(1)

    def get_votes(self, pose, mean_poses, size):
        pool_positions = self.pool_positions.view(1, 1, 1, self.pool_size, 1,
                                                  2)

        pool_positions = pool_positions.view(1, 1, 1, self.pool_size, 1, 2, 1)

        mean_inv_pose = inverse_pose(mean_poses)
        pose_trans = posevec_to_rotmat(mean_inv_pose).unsqueeze(-4)

        transformed_positions = torch.matmul(pose_trans, pool_positions)
        transformed_positions = transformed_positions.view(
            -1, self.in_channels, 2)


        transformed_positions = transformed_positions.detach()
        theta = self.theta_gen2(
            self.theta_gen1(
                transformed_positions[:, :, :].contiguous().view(
                    -1, self.in_channels * 2)))
        theta = theta.view(*size, self.pool_size, self.out_channels,
                           self.in_channels)

        theta = theta.transpose(-2, -3)
        sin, cos = theta.sin(), theta.cos()

        g = torch.stack([cos, -sin, sin, cos], dim=-1)
        g = g.view(*list(g.size())[:-1], 2, 2)

        pose = pose.view(*size, 1, self.pool_size, self.in_channels, 2)

        pose_mat = posevec_to_rotmat(pose)
        pose_mat = torch.matmul(pose_mat, g)
        pose = pose_mat[..., 0].contiguous().view(
            *size, self.out_channels, self.pool_size, self.in_channels, 2)

        pose = pose.view(*size, self.out_channels,
                         self.pool_size * self.in_channels, 2)

        return pose

    def mean(self, vote, weight):
        weight = weight.unsqueeze(-1)
        mean = (vote * weight).sum(dim=-2)
        mask = torch.sign(torch.abs(mean))
        mean = mean / ((mean.norm(p=2, dim=-1).unsqueeze(-1)) + (1 - mask))
        return mean

    def mean_per_channel(self, pose):
        mean = pose.sum(dim=3)
        mask = torch.sign(mean.norm(p=2, dim=-1).unsqueeze(-1))
        mean = mean / (mean.norm(p=2, dim=-1).unsqueeze(-1) + (1 - mask))
        return mean

    def mean_per_channel_weighted(self, pose, weight):
        weight = weight.unsqueeze(-1)
        mean = (pose*weight).sum(dim=3)
        mask = torch.sign(mean.norm(p=2, dim=-1).unsqueeze(-1))
        mean = mean / (mean.norm(p=2, dim=-1).unsqueeze(-1) + (1 - mask))
        return mean

    def distance(self, pose, vote):
        pose = pose[:, :, :, :, None, :].expand_as(vote)
        distance = (pose * vote).sum(dim=-1)
        return (-distance + 1) / 2

    def forward(self, x, a, pose, size):
        pooled_size = (size[0], int((size[1] + 1) / (self.pool_length)),
                       int((size[2] + 1) / (self.pool_length)))

        a = a.view(*size, self.in_channels)
        pose = pose.view(*size, self.in_channels, 2)

        a = space_to_depth(a.unsqueeze(-1), self.pool_length).squeeze(-1)
        pose = space_to_depth(pose, self.pool_length)

        pose = pose.view(*pooled_size, self.pool_size, self.in_channels, 2)
        a = a.view(*pooled_size, self.pool_size, self.in_channels)


        means = self.mean_per_channel(pose)

        vote = self.get_votes(pose, means, pooled_size)

        a = a.view(*pooled_size, self.pool_size * self.in_channels)
        a = a[:, :, :, None, :].expand(-1, -1, -1, self.out_channels, -1)
        weight = a

        pose = self.mean(vote, weight)

        beta = self.beta.view(1, 1, 1, -1)
        alpha = self.alpha.view(1, 1, 1, -1)

        for r in range(self.num_iterations):
            distance = self.distance(pose, vote)
            w = (1 - distance)
            weight = w * a
            pose = self.mean(vote, weight)

        neg_distance = (1 - self.distance(pose, vote)) * weight
        w_sum = torch.sign(torch.abs(weight)).sum(dim=-1)
        mask = torch.sign(torch.abs(w_sum))
        neg_distance = (neg_distance.sum(dim=-1) / (w_sum + (1 - mask))) * mask

        agreement = torch.sigmoid(alpha * neg_distance + (beta - 1)) * mask

        x = space_to_depth(x.view(*size, -1, 1), self.pool_length).squeeze(-1)
        x = x.view(*pooled_size, self.pool_size, -1).max(dim=-2)[0]
        x = x.view(-1, self.f_in_size)
        x_out = self.lin(x)

        return x_out, agreement, pose, x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
