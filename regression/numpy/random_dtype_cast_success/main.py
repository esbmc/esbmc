import numpy as np

x = np.random.randint(0, 128, dtype=np.int8)

assert x >= 0
assert x <= 127
