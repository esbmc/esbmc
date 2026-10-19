import numpy as np


def pick(k: int) -> int:
    a = np.random.choice([1, 2], size=k)
    return a[0]


pick(2)
