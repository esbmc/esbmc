import numpy as np

x = np.random.randint(0, 256, dtype=np.uint8)

assert x >= 0
assert x <= 255
