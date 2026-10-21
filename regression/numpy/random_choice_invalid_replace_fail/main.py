import numpy as np


def pick(replace: bool) -> int:
    return np.random.choice([1, 2], replace=replace)


pick(True)
