import os.path as osp

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from group_capsules.datasets import AffNIST
from group_capsules.nn import (ImageGradient, GroupConvLayer,
                               PoolingCapsuleLayer)
from group_capsules.utils import grid, make_batch, spread_loss
import numpy as np
from group_capsules.utils.image_gradients import compute_image_gradients

BATCH_SIZE = 48
LEARNING_RATE = 0.001
HEIGHT = 32
WIDTH = 32


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.grad = ImageGradient()
        self.conv1 = GroupConvLayer(1, 16, kernel_size=5, initial=True)
        self.pool1 = PoolingCapsuleLayer(1, 16, num_iterations=1, f_in_size=16)
        self.conv2 = GroupConvLayer(16, 16, kernel_size=5)
        self.pool2 = PoolingCapsuleLayer(16, 32, num_iterations=1)
        self.conv3 = GroupConvLayer(32, 32, kernel_size=5)
        self.pool3 = PoolingCapsuleLayer(32, 32, num_iterations=1)
        self.conv4 = GroupConvLayer(32, 32, kernel_size=5)
        self.pool4 = PoolingCapsuleLayer(32, 64, num_iterations=1)
        self.conv5 = GroupConvLayer(64, 64, kernel_size=5)
        self.pool5 = PoolingCapsuleLayer(64, 10, num_iterations=1)

        self.e1, self.p1 = grid(HEIGHT, WIDTH, device=device)
        self.e2, self.p2 = grid(HEIGHT // 2, WIDTH // 2, device=device)
        self.e3, self.p3 = grid(HEIGHT // 4, WIDTH // 4, device=device)
        self.e4, self.p4 = grid(HEIGHT // 8, WIDTH // 8, device=device)
        self.e5, self.p5 = grid(HEIGHT // 16, WIDTH // 16, device=device)

    def forward(self, img_batch):
        img_batch = img_batch.permute(0, 2, 3, 1).contiguous()
        batch_size, height, width, in_channels = img_batch.size()

        pose, act = compute_image_gradients(img_batch)
        x = img_batch.view(-1, in_channels)


        be1, bp1 = make_batch(batch_size, self.e1, self.p1)
        x = F.relu(self.conv1(x, be1, bp1, pose))
        size = batch_size, height, width
        x, act, pose, _ = self.pool1(x, F.relu(act), pose, size)

        be2, bp2 = make_batch(batch_size, self.e2, self.p2)
        x = F.relu(self.conv2(x, be2, bp2, pose))
        size = batch_size, height // 2, width // 2
        x, act, pose, _ = self.pool2(x, F.relu(act), pose, size)

        be3, bp3 = make_batch(batch_size, self.e3, self.p3)
        x = F.relu(self.conv3(x, be3, bp3, pose))
        size = batch_size, height // 4, width // 4
        x, act, pose, _ = self.pool3(x, F.relu(act), pose, size)

        be4, bp4 = make_batch(batch_size, self.e4, self.p4)
        x = F.relu(self.conv4(x, be4, bp4, pose))
        size = batch_size, height // 8, width // 8
        x, act, pose, _ = self.pool4(x, F.relu(act), pose, size)

        be5, bp5 = make_batch(batch_size, self.e5, self.p5)
        x = F.relu(self.conv5(x, be5, bp5, pose))
        size = batch_size, height // 16, width // 16
        x, act, pose, _ = self.pool5(x, F.relu(act), pose, size)


        return F.log_softmax(x, dim=1), act.squeeze(1).squeeze(1)


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
train_path = osp.join(path, 'MNIST')
test_path = osp.join(path, 'AffNIST')
train_transform = T.Compose(
    [T.Pad(10),
     T.RandomAffine(0, translate=(0.15, 0.15)),
     T.ToTensor()])
test_transform = T.Compose([T.Pad(4), T.ToTensor()])
train_dataset = MNIST(train_path, train=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = AffNIST(test_path, train=False, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(epoch):
    model.train()
    a_loss_sum = 0
    f_loss_sum = 0
    loss_count = 0

    if epoch == 21:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    if epoch == 31:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    if epoch == 41:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00005

    for i, (img_batch, target) in enumerate(train_loader):
        img_batch, target = img_batch.to(device), target.to(device)
        optimizer.zero_grad()
        out, act = model(img_batch)
        out_loss = F.nll_loss(out, target)
        act_loss = spread_loss(act, target, epoch, device=device)
        loss = out_loss + act_loss
        loss.backward()

        a_loss_sum += act_loss.item()
        f_loss_sum += out_loss.item()
        loss_count += 1
        optimizer.step()

        if loss_count == 20:
            print('Step {}/{}, CapsuleLoss: {}, CNNLoss: {}'
                  .format(i+1,len(train_loader), a_loss_sum / loss_count,
                          f_loss_sum / loss_count))
            loss_count = 0
            f_loss_sum = 0
            a_loss_sum = 0

    #torch.save(model.state_dict(),
    #          'models/model_affmnist/e{}.model'.format(epoch))

def test(epoch, loader, dataset):
    model.eval()

    correct_f = 0
    correct_a = 0

    for (img_batch, target) in loader:
        img_batch, target = img_batch.to(device), target.to(device)
        f_out, a_out = model(img_batch.to(device))
        f_pred = f_out.max(1)[1]
        a_pred = a_out.max(1)[1]

        eq = a_pred.eq(target)
        correct_a += eq.sum()

        eq = f_pred.eq(target)
        correct_f += eq.sum()

    np.set_printoptions(precision=4)
    correct_a = correct_a.item()
    print('Epoch:', epoch, 'Capsule', 'Accuracy:',
         correct_a / len(dataset))

    correct_f = correct_f.item()
    print('Epoch:', epoch, 'Conv', 'Accuracy:', correct_f / len(dataset))

    return correct_f / len(dataset), correct_a / len(dataset)


for epoch in range(1, 46):
    train(epoch)
    print('train accuracy')
    test(epoch, train_loader, train_dataset)
    if epoch == 1 or epoch == 25 or epoch == 45:
        print('test accuracy')
        test(epoch, test_loader, test_dataset)
    else:
        print('Epoch: {:02d}'.format(epoch))
