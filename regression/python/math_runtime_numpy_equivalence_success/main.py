import math
import numpy as np

x = nondet_float()
if x == 0.0:
    assert math.copysign(1.0, math.sinh(x)) == math.copysign(1.0, np.sinh(x))
    assert math.cos(x) == np.cos(x)
if x == float("inf"):
    assert math.exp(-x) == np.exp(-x)
