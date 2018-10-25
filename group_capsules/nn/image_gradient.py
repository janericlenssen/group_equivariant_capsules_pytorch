import torch
from torch.nn import Module
from torch.nn.functional import conv2d


class ImageGradient(Module):
    def __init__(self):
        super(ImageGradient, self).__init__()

        filter_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.register_buffer('filter_x', filter_x.view(1, 1, 3, 3))

        filter_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.register_buffer('filter_y', filter_y.view(1, 1, 3, 3))

    def forward(self, img_batch):  # batch_size x 1 x height x width
        assert img_batch.size(1) == 1
        batch_size, _, height, width = img_batch.size()

        pose_x = conv2d(img_batch, self._buffers['filter_x'], padding=1)
        pose_y = conv2d(img_batch, self._buffers['filter_y'], padding=1)
        pose = torch.cat([pose_x, pose_y], dim=1)

        act = pose.norm(dim=1, keepdim=True)
        mask = (act > 0).expand_as(pose)
        pose[mask.expand_as(pose)] /= act.expand_as(pose)[mask]

        #  act: batch_size x 1 x height x width
        # pose: batch_size x 2 x height x width
        return act, pose

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
