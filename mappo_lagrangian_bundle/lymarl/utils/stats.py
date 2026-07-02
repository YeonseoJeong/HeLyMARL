import numpy as np
from typing import Tuple


def moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average for a 1-D array. Returns same-length output."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x

    out = np.zeros_like(x, dtype=np.float32)
    csum = 0.0
    for i in range(len(x)):
        csum += float(x[i])
        if i >= window:
            csum -= float(x[i - window])
            out[i] = csum / float(window)
        else:
            out[i] = csum / float(i + 1)
    return out


def block_avg_1d(x: np.ndarray, block: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Block-average a 1-D array.
    Returns (xs, ys) where xs[i] is the end step of block i and ys[i] is its mean.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    T = x.shape[0]

    if T == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)

    xs, ys = [], []
    for start in range(0, T, block):
        chunk = x[start:start + block]
        if chunk.size == 0:
            continue
        xs.append(start + chunk.size)
        ys.append(float(chunk.mean()))

    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.float32)