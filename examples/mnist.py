import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.nn import Linear
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import MNIST
import torchvision.transforms as T
from group_capsules.utils import grid, grid_cluster, make_batch, spread_loss
from group_capsules.nn import GroupConvLayer, PoolingCapsuleLayer
from group_capsules.utils.image_gradients import compute_image_gradients
#from group_capsules.utils.evaluate_poses import evaluate_poses


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


        self.recon1 = Linear(64 + 20 + 10, 512)
        self.recon2 = Linear(512, 1024)
        self.recon3 = Linear(1024, 784)


    def forward(self, img_batch):

        batch_size, height, width, in_channels = img_batch.size()
        pose, a = compute_image_gradients(img_batch)

        pose = pose.detach()

        a = a.detach()
        x = img_batch.view(-1, in_channels)
        size = batch_size, height, width

        be1, bp1, bc1 = make_batch(batch_size, e1, p1, c1)

        x = F.relu(self.conv1(x, be1, bp1, pose))

        x, a, pose, _ = self.pool1(x, F.relu(a), pose, size)
        size = batch_size, 16, 16
        be2, bp2, bc2 = make_batch(batch_size, e2, p2, c2)

        x = F.relu(self.conv2(x, be2, bp2, pose)) * a.view(-1, a.size()[-1])

        x, a, pose, _ = self.pool2(x, F.relu(a), pose, size)
        size = batch_size, 8, 8
        be3, bp3, bc3 = make_batch(batch_size, e3, p3, c3)

        x = F.relu(self.conv3(x, be3, bp3, pose)) * a.view(-1, a.size()[-1])

        x, a, pose, _ = self.pool3(x, F.relu(a), pose, size)
        size = batch_size, 4, 4
        be4, bp4, bc4 = make_batch(batch_size, e4, p4, c4)

        x = F.relu(self.conv4(x, be4, bp4, pose)) * a.view(-1, a.size()[-1])

        x, a, pose, _ = self.pool4(x, F.relu(a), pose, size)
        size = batch_size, 2, 2
        be5, bp5, bc5 = make_batch(batch_size, e5, p5, c5)

        x = F.relu(self.conv5(x, be5, bp5, pose)) * a.view(-1, a.size()[-1])

        x, a, pose, for_recon = self.pool5(x, F.relu(a), pose, size)

        rec_in = torch.cat(
            [for_recon.view(-1, 64), a.view(-1, 10), pose.view(-1, 20)], dim=1)
        rec = F.relu(self.recon1(rec_in))
        rec = F.relu(self.recon2(rec))
        rec = torch.sigmoid(self.recon3(rec))

        return F.log_softmax(x, dim=1), a.squeeze(1).squeeze(1), pose, rec


model = Net()
#model.load_state_dict(torch.load('models/model_mnist/e50.model'))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    a_loss_sum = 0
    f_loss_sum = 0
    loss_count = 0
    r_loss_sum = 0

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
        img_batch_28, target = img_batch.to(device), target.to(device)
        img_batch = torch.nn.ZeroPad2d(2)(img_batch_28)
        img_batch = img_batch.squeeze(1).unsqueeze(-1)
        optimizer.zero_grad()

        f_out, a_out, _, img_out = model(img_batch.to(device))

        a_loss = spread_loss(a_out, target, epoch, device=device)
        f_loss = F.nll_loss(f_out, target)
        recon_loss = (
            torch.abs(img_out.view(-1, 28, 28) - img_batch_28.view(-1, 28, 28))) \
            .sum(dim=2).sum(dim=1).mean(dim=0)

        loss = a_loss + f_loss + 0.01 * recon_loss

        loss.backward()

        a_loss_sum += a_loss.item()
        f_loss_sum += f_loss.item()
        r_loss_sum += recon_loss.item()
        loss_count += 1
        optimizer.step()

        if loss_count == 100:
            print('Step {}/{}, CapsuleLoss: {}, CNNLoss: {}, ReconLoss: {}'
                  .format(i+1,len(train_loader),a_loss_sum / loss_count,
                          f_loss_sum / loss_count, r_loss_sum / loss_count))
            loss_count = 0
            f_loss_sum = 0
            a_loss_sum = 0
            r_loss_sum = 0

    #torch.save(model.state_dict(),
    #          'models/model_mnist/e{}.model'.format(epoch))

def test(epoch, loader):
    model.eval()

    correct_f = 0
    correct_a = 0

    for (img_batch, target) in loader:
        img_batch, target = img_batch.to(device), target.to(device)
        img_batch = torch.nn.ZeroPad2d(2)(img_batch)
        img_batch = img_batch.squeeze(1).unsqueeze(-1)
        f_out, a_out, _, _ = model(img_batch.to(device))
        f_pred = f_out.max(1)[1]
        a_pred = a_out.max(1)[1]

        eq = a_pred.eq(target)
        correct_a += eq.sum()

        eq = f_pred.eq(target)
        correct_f += eq.sum()

    np.set_printoptions(precision=4)
    correct_a = correct_a.item()
    print('Epoch:', epoch, 'Capsule', 'Accuracy:',
         correct_a / len(test_dataset))

    correct_f = correct_f.item()
    print('Epoch:', epoch, 'Conv', 'Accuracy:', correct_f / len(test_dataset))

    return correct_f / len(test_dataset), correct_a / len(test_dataset)


for epoch in range(1, 46):
    train(epoch)
    print('test_accuracy')
    f_test_acc, a_test_acc = test(epoch, test_rot_loader)
    #if epoch % 5 == 0:
    #    evaluate_poses(model, train_loader, test_loader, device)
