import numpy as np

x = nondet_float()
if x == 0.0:
    assert np.copysign(1.0, np.tanh(x)) == np.copysign(1.0, x)
    assert np.copysign(1.0, np.arcsin(x)) == np.copysign(1.0, x)
    assert np.exp(x) == 1.0
