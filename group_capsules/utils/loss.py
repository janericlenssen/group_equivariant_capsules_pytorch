import torch
import torch.nn.functional as F
from group_capsules.utils import one_hot


def spread_loss(x, target, epoch, device=None):
    target = one_hot(target, x.size(1), device=device)
    m = min(0.1 + 0.1 * epoch, 0.9)
    m = torch.tensor(m, dtype=target.dtype, device=device)
    act_t = (x * target).sum(dim=1)
    loss = ((F.relu(m - (act_t.view(-1, 1) - x))**2) * (1 - target))
    loss = loss.sum(1).mean()
    return loss
