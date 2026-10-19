import numpy as np


def make():
    return np.full((2, ), "hi")


s = make()
assert s[1] == "hi" and len(s[0]) == 2
