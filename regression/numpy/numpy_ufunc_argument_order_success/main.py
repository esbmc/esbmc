import math
import numpy as np

x = 1.0
calls = 0


def flip() -> float:
    global x
    x = -1.0
    return 1.0


def arg() -> float:
    global calls
    calls = calls + 1
    return 3.0


r = np.arctan2(x, flip())
assert r > 0.0
x = 1.0
assert math.atan2(x, flip()) > 0.0
h = np.hypot(arg(), [4.0, 4.0])
assert calls == 1
assert h[0] >= 4.0 and h[1] <= 5.7
