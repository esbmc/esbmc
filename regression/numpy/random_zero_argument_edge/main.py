import numpy as np

e = np.random.exponential()
k = np.random.poisson()
u = np.random.uniform()
n = np.random.normal()

assert e >= 0.0
assert k >= 0
assert u >= 0.0
assert u <= 1.0
assert n == n
