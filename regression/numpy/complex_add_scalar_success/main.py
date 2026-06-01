import numpy as np

z = np.conjugate(1 + 2j)
assert z.real == 1.0
assert z.imag == -2.0
