import numpy as np


def draw(scale: float) -> float:
    return np.random.exponential(scale)


def pick() -> int:
    return np.random.choice([2, 4])


x = draw(2.0)
k = pick()

assert x >= 0.0
assert k == 2 or k == 4
