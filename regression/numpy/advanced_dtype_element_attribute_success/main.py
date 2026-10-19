import numpy as np

a = np.array([2j, 1 + 2j])
b = np.array([1.5, 2.5])
x = a[1]
y = b[1]
assert x.real == 1.0 and x.imag == 2.0
assert y.real == 2.5 and y.imag == 0.0
