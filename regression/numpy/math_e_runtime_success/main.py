import math
import numpy as np

x = nondet_float()
if x >= 1.0 and x <= 2.0:
    y = np.exp(x)
    z = math.exp(x)
    assert y >= 2.71 and y < 7.39 and z >= 2.71 and z < 7.39
