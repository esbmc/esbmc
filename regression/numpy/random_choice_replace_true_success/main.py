import numpy as np

x = np.random.choice((2, 4), replace=True)

assert x % 2 == 0
