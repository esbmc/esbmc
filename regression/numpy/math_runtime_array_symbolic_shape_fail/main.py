import numpy as np

n = nondet_int()
a = np.array([0.5, 1.0, 2.0])
if n >= 0 and n <= 3:
    v = a[:n]
    b = np.exp(v)
