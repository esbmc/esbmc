import numpy as np

x = nondet_float()
if x == 0.0:
    a = np.array([x, 1.0])
    t = np.tanh(a)
    assert np.copysign(1.0, t[0]) == np.copysign(1.0, x)
