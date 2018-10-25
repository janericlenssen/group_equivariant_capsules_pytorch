from collections import Iterable
from itertools import repeat


def repeat_to(src, dims):
    out = src if isinstance(src, Iterable) else [src]
    assert len(out) <= dims

    if len(out) < dims:
        out += repeat(out[-1], dims - len(out))
    return out
