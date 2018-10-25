import torch
import numpy as np
import torch.nn as nn

def compute_image_gradients(images):
    a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    images = images.view(-1, 1, images.size()[1], images.size()[2])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).cuda())

    G_x = conv1(images)

    b = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0).cuda())
    G_y=conv2(images)

    length = torch.sqrt(torch.pow(G_x,2) + torch.pow(G_y,2))
    # -G_y for inverse pose
    v = torch.cat([-G_x, -G_y], dim=1)
    mask = torch.zeros_like(length)
    mask[length==0] = 1

    v = v/(length+mask)
    v = v.transpose(1,3).transpose(1,2).contiguous().view(-1, 1, 2)
    length = length.view(-1, 1)
    return v, length