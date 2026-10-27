import numpy as np

x = nondet_float()
if x >= 0.9 and x <= 1.1:
    assert np.sin(x) != 0.8414709848078965
