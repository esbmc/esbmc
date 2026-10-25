import math
import numpy as np

calls = 0


def bump() -> float:
    global calls
    calls += 1
    return 0.5


y = np.exp(bump())
z = math.exp(bump())
assert calls == 2
assert y > 1.1 and z > 1.1
