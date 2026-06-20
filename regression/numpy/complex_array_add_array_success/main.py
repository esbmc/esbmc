import numpy as np

z = np.conjugate(2 + 5j)
assert z.real == 2.0
assert z.imag == -5.0
