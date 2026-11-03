import numpy as np

a = np.random.random_sample()
b = np.random.ranf()
c = np.random.randn()

assert 0.0 <= a < 1.0
assert 0.0 <= b < 1.0
assert c - c == 0.0
