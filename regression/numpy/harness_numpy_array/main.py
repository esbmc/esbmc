# Verification harness for numpy array construction
# (src/python-frontend/models/numpy.py).
#
# zeros(n)/ones(n) build length-n float lists filled with 0.0/1.0; array(list)
# wraps a list. The numpy frontend handler supports constant inputs, so this
# harness is concrete.
#
# ENSURES:
#   E1: zeros(n) has length n and every element is 0.0
#   E2: ones(n) has length n and every element is 1.0
#   E3: array(list) preserves length and element order
import numpy as np

z = np.zeros(3)
assert len(z) == 3
assert z[0] == 0.0 and z[1] == 0.0 and z[2] == 0.0  # E1

o = np.ones(2)
assert len(o) == 2
assert o[0] == 1.0 and o[1] == 1.0  # E2

a = np.array([10, 20, 30])
assert len(a) == 3
assert a[0] == 10 and a[1] == 20 and a[2] == 30  # E3
