import numpy as np

z = np.conjugate(5 + 6j)
assert z.real == 5.0
assert z.imag == -6.0
