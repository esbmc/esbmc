import math
import numpy as np

calls = 0


def f() -> float:
    global calls
    calls += 1
    return 0.5


y = math.exp(np.sin(f()))
assert calls == 1 and y > 1.0 and y < 2.72
