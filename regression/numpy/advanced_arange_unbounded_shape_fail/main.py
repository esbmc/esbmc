import numpy as np


def ramp(n: int):
    return np.arange(stop=n)


a = ramp(3)
b = ramp(5)
assert len(a) == 3 and len(b) == 5
