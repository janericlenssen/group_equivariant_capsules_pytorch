import torch
from torch_unique import unique


def grid(height, width, dtype=None, device=None):
    edge_index = grid_index(height, width, device)
    pos = grid_pos(height, width, dtype, device)
    return edge_index, pos


def grid_index(height, width, device=None):
    w = width
    kernel = [-w - 1, -1, w - 1, -w, 0, w, -w + 1, 1, w + 1]
    kernel = torch.tensor(kernel, device=device)

    row = torch.arange(height * width, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, kernel.size(0))
    col = row + kernel.view(1, -1)
    row, col = row.view(height, -1), col.view(height, -1)
    index = torch.arange(3, row.size(1) - 3, dtype=torch.long, device=device)
    row, col = row[:, index].view(-1), col[:, index].view(-1)

    mask = (col >= 0) & (col < height * width)
    row, col = row[mask], col[mask]

    edge_index = torch.stack([row, col], dim=0)
    edge_index = coalesce(edge_index, height * width)

    return edge_index


def grid_pos(height, width, dtype=None, device=None):
    x = torch.arange(width, dtype=dtype, device=device)
    y = (height - 1) - torch.arange(height, dtype=dtype, device=device)

    x = x.repeat(height)
    y = y.unsqueeze(-1).repeat(1, width).view(-1)

    return torch.stack([x, y], dim=-1)


def coalesce(edge_index, num_nodes=None):
    num_nodes = edge_index.max().item() + 1 if num_nodes is None else num_nodes
    row, col = edge_index

    index = num_nodes * row + col
    perm = torch.arange(index.size(0), out=row.new())
    _, perm = unique(index)
    edge_index = edge_index[:, perm]

    return edge_index


def grid_cluster(height, width, size, device=None):
    num_row, num_col = int((height + 1) / size), int((width + 1) / size)

    col = torch.arange(num_col, dtype=torch.long, device=device)
    col = col.view(-1, 1).repeat(1, size).view(-1)[:width]

    cluster = torch.stack([col for _ in range(height)], dim=0)

    row = torch.arange(num_row, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, size).view(-1)[:height]
    row = (num_col * row).view(-1, 1)

    cluster = (col + row).view(-1)
    return cluster


def pool_grid(pool_size):
    if pool_size == 4:
        return torch.FloatTensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0],
                                  [1.0, 1.0]]).cuda()
    elif pool_size == 9:
        return torch.FloatTensor([[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
                                  [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
                                  [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]]).cuda()
    elif pool_size == 16:
        return torch.FloatTensor([[-3, -3], [-1, -3], [1, -3], [3, -3], [
            -3, -1
        ], [-1, -1], [1, -1], [3, -1], [-3, 1], [-1, 1], [1, 1], [3, 1],
                                  [-3, 3], [-1, 3], [1, 3], [3, 3]]).cuda()
    elif pool_size == 25:
        return torch.FloatTensor(
            [[-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2], [-2, -1], [-1, -1],
             [0, -1], [1, -1], [2, -1], [-2, 0], [-1, 0], [0, 0], [1, 0],
             [2, 0], [-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1], [-2, 2],
             [-1, 2], [0, 2], [1, 2], [2, 2]]).cuda()
    elif pool_size == 64:
        return torch.FloatTensor([[(x - 3.5) / 4, (y - 3.5) / 4]
                                  for y in range(8) for x in range(8)]).cuda()
    else:
        raise NotImplementedError
