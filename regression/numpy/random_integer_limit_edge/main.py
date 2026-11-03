import numpy as np

k = np.random.binomial(9223372036854775807, 0.0)
v = np.random.randint(-9223372036854775808, 9223372036854775807)

assert k == 0
assert v < 9223372036854775807
