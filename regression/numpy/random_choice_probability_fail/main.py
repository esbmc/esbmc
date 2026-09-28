import numpy as np

x = np.random.choice([1, 10], p=[0.5, 0.5])

assert x == 1 or x == 10
