def pseudo_cartesian(edge_index, pos, size=None):
    row, col = edge_index
    cartesian = (pos[col] - pos[row]).float()
    cartesian /= 2 * (cartesian.abs().max() if size is None else size)
    cartesian += 0.5

    return cartesian
