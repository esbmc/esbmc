import numpy as np

a = np.random.choice([1, 2], size=(64, 64))

assert a.size == 4096
