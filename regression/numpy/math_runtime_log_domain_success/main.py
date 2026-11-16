import numpy as np

x = nondet_float()
if x < 0.0:
    y = np.log10(x)
    assert y != y
if x == 0.0:
    assert np.log(x) == -float("inf")
