import numpy as np


def pick(xs: list[int]) -> int:
    return np.random.choice(xs)


pick([1, 2])
