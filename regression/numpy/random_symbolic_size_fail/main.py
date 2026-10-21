import numpy as np


def sample(n: int) -> float:
    a = np.random.random(n)
    return a[0]


sample(2)
