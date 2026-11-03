import numpy as np

cond = nondet_bool()
a = np.zeros((2, 2))
b = np.zeros(3)
if cond:
    a = b

assert a.size == 3
