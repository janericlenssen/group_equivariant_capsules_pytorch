import torch
import random
from PIL import Image
import torchvision.transforms as T
import math


def evaluate_poses(model, train_loader, test_loader, device,
                   histograms=False):
    poses = []
    rot_poses = []
    rots = []

    targets = []
    for bla, (img_batch, target) in enumerate(test_loader):
        targets.append(target)
    targets = torch.cat(targets, dim=0)


    for bla, (img_batch, target) in enumerate(test_loader):
        img_batch, target = img_batch.to(device), target.to(device)
        img_batch = torch.nn.ZeroPad2d(2)(img_batch)
        img_batch = img_batch.squeeze(1).unsqueeze(-1)
        _, _, pose, _ = model(img_batch.to(device))
        pose = pose.squeeze(1).squeeze(1).cpu()

        rot_img_batch = []

        for i in range(target.size(0)):
            poses.append(pose[i, target[i]].detach().cpu())

            rand = random.random() * 360
            rots.append(rand)
            img = Image.fromarray(
                img_batch[i].cpu().squeeze().numpy() * 255, mode='F')
            img = img.rotate(rand, Image.BILINEAR)
            img = T.ToTensor()(img) / 255.0
            rot_img_batch.append(img)

        img_batch = torch.stack(rot_img_batch, dim=0).to(device)
        img_batch = img_batch.squeeze(1).unsqueeze(-1)
        _, _, pose, _ = model(img_batch.to(device))
        pose = pose.squeeze(1).squeeze(1).cpu()

        for i in range(target.size(0)):
            rot_poses.append(pose[i, target[i]].detach().cpu())

    poses = torch.stack(poses, dim=0)
    rot_poses = torch.stack(rot_poses, dim=0)
    rots = torch.tensor(rots)


    thetas = []
    for i in range(poses.size(0)):
        pose = poses[i]
        rot_pose = rot_poses[i]
        rot = rots[i]
        rot /= 180.0
        rot *= math.pi
        a = torch.tensor(
            [math.cos(rot), math.sin(rot),
             -math.sin(rot), math.cos(rot)])

        corrected_pose = torch.mm(a.view(2, 2), pose.view(-1, 1)).squeeze()

        dot_product = (rot_pose * corrected_pose).sum()
        theta = 180 * torch.acos(dot_product.clamp(min=-1, max=1)) / math.pi
        thetas.append(theta.item())


    avg_error_global = 0
    thetas = torch.tensor(thetas)
    for target in range(10):

        thetas_c = thetas[targets == target]
        summ = (targets == target).sum().float()

        avg_error = thetas_c.sum()
        avg_error_global += avg_error

        print('Avg. pose error class {}: {}'.format(target, avg_error/summ))


        if histograms:
            bins = torch.arange(5, 185, 5)

            a = []

            for b in bins:
                smaller = (thetas_c <= b.float())
                x = smaller.sum().float()
                x = x / summ.float()
                a.append(x.item())

            cur = 0
            for i in range(len(a)):
                bla = a[i]
                a[i] -= cur
                cur = bla

            txt = str(target) + ':\n'

            for i in range(len(a)):
                txt += '({},{})\n'.format(bins[i] - 5, a[i])

            print(txt)
    print('Global avg. pose error: {}'.format(avg_error_global/10000))