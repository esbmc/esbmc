import numpy as np

a = np.array([10, 20, 30, 40, 50])
c = nondet_bool()
if c:
    v = a[2:2]
else:
    v = a[1:3]
assert len(v) == 0 or len(v) == 2
