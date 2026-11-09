import math
import numpy as np

calls = 0


def arg() -> float:
    global calls
    calls += 1
    return 0.5


a = np.expm1(arg())
b = math.log1p(arg())
c = np.hypot(arg(), 1.0)
d = math.atan2(arg(), 1.0)
assert calls == 4
