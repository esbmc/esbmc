import numpy as np

z = np.array([1j, 2j, 3j])
v = z[::2]
c = np.random.choice(v)

assert c.imag == 1.0 or c.imag == 3.0
