import os.path as osp
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
from group_capsules.utils import grid, grid_cluster
from group_capsules.utils.image_gradients import compute_image_gradients
from group_capsules.utils.evaluate_poses import evaluate_poses


batch_size = 32
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
train_dataset = MNIST(path, train=True, transform=T.Compose([T.RandomRotation(
    180, resample=Image.BILINEAR), T.ToTensor()]), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNIST(path, train=False, transform=T.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size)
test_rot_dataset = MNIST(
    path,
    train=False,
    transform=T.Compose([T.RandomRotation(180, resample=Image.BILINEAR),
                         T.ToTensor()]))
test_rot_loader = DataLoader(test_rot_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(e1, p1), c1 = grid(32, 32, device=device), grid_cluster(32, 32, 2, device)
(e2, p2), c2 = grid(16, 16, device=device), grid_cluster(16, 16, 2, device)
(e3, p3), c3 = grid(8, 8, device=device), grid_cluster(8, 8, 2, device)
(e4, p4), c4 = grid(4, 4, device=device), grid_cluster(4, 4, 2, device)
(e5, p5), c5 = grid(2, 2, device=device), grid_cluster(2, 2, 2, device)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, img_batch):
        pose, a = compute_image_gradients(img_batch)

        pose = pose.detach()
        a = a.detach()


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

        def mean(vote):
            mean = vote.sum(dim=-3)
            mask = torch.sign(torch.abs(mean))
            mean = mean / ((mean.norm(p=2, dim=-1).unsqueeze(-1)) + (1 - mask))
            return mean

        x = 32
        for i in range(5):
            pose = pose.view(-1, x, x, 1, 2)
            a = a.view(-1, x, x, 1, 1)

            a = space_to_depth(a, 2)
            pose = space_to_depth(pose, 2)

            pose = pose.view(-1, int(x/2), int(x/2), 4, 1, 2)
            a = a.view(-1, int(x/2), int(x/2), 4, 1, 1)
            pose = mean(pose)
            x = int(x/2)

        mean_pose = pose.view(-1, 1, 2).expand(-1, 10, -1)
        return None, None, mean_pose, None


model = Net()
model.to(device)

evaluate_poses(model, train_loader, test_loader, device)
