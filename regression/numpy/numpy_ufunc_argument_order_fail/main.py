import numpy as np

x = 1.0


def flip() -> float:
    global x
    x = -1.0
    return 1.0


r = np.arctan2(x, flip())
assert r < 0.0
