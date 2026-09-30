import numpy as np

cond = nondet_bool()
if cond:
    a = np.zeros((2, 2))
else:
    a = np.zeros(3)

assert a.size == 3
