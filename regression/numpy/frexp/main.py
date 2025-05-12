import numpy as np

m, e = np.frexp(8.0)
assert m * (2 ** e) >= 8.0
m, e = np.frexp(0.0)
assert m == 0.0 and e == 0
