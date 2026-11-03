import numpy as np

z = np.random.choice([1 + 2j, 3j])

assert z.imag == 2.0 or z.imag == 3.0
assert z.real == 1.0 or z.real == 0.0
