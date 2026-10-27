import numpy as np

x = nondet_float()
if x == float("inf"):
    assert np.exp(-x) == 0.0
    assert np.tanh(x) == 1.0
    s = np.sin(x)
    assert s != s
