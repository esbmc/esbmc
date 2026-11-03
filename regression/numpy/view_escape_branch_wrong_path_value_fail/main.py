import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
c = nondet_bool()
if c:
    v = a[0]
else:
    v = a[1]
if c:
    assert v[2] == 3
else:
    # The else path reads the second row.
    assert v[2] == 3
