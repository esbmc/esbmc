import numpy as np

a = np.array([10, 20, 30])
x = np.random.choice(a)

assert x == 10 or x == 20 or x == 30
