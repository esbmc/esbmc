import numpy as np

c = nondet_bool()
x = np.array([0.0, 1.0])
o = np.full(2, 5.0)
np.exp(x, where=c, out=o)
assert o[0] == 1.0 or o[0] == 5.0
if not c:
    assert o[1] == 5.0
if c:
    assert o[1] > 2.71 and o[1] < 2.72
