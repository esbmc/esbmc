import numpy as np


def sample(rows: int) -> float:
    a = np.random.normal(size=(rows, 2))
    return a[0][0]


sample(2)
