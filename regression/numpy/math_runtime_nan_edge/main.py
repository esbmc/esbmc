import numpy as np

x = nondet_float()
if x != x:
    e = np.exp(x)
    t = np.arctan(x)
    assert e != e and t != t
