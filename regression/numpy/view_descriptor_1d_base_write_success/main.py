import numpy as np

# A live 1-D slice view must observe writes to the base array through the
# shared buffer, not a stale private copy taken at view-creation time.
a = np.array([1, 2, 3, 4])
v = a[1:3]
a[1] = 99

assert v[0] == 99
assert a[1] == 99
