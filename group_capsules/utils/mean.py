def weighted_mean(pose, weight=None):
    if weight is None:
        mean = pose.mean(dim=-2)
    else:
        mean = (pose * weight).sum(dim=-2) / weight.sum(dim=-1)

    return normalize(mean)


def normalize(pose):
    return pose / pose.norm(p=2, dim=-1)
