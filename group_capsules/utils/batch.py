from itertools import repeat

import torch


def make_batch(batch_size, edge_index, pos, cluster=None):
    num_nodes = pos.size(0)

    edge_index = torch.stack(tuple(repeat(edge_index, batch_size)), dim=1)
    idx = torch.arange(batch_size, dtype=torch.long, device=edge_index.device)
    edge_index += num_nodes * idx.view(1, -1, 1)
    edge_index = edge_index.view(2, -1)

    pos = torch.cat(tuple(repeat(pos, batch_size)), dim=0)

    if cluster is not None:
        max_value = cluster.max() + 1
        cluster = torch.stack(tuple(repeat(cluster, batch_size)), dim=0)
        tmp = torch.arange(
            batch_size, dtype=cluster.dtype, device=cluster.device)
        tmp *= max_value
        cluster += tmp.view(-1, 1)
        cluster = cluster.view(-1)

        return edge_index, pos, cluster
    else:
        return edge_index, pos
