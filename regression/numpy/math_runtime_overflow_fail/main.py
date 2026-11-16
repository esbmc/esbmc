import numpy as np

x = nondet_float()
if x > 710.0 and x < float("inf"):
    y = np.cosh(x)
    assert y < float("inf"), "overflow"
