import numpy as np

p = np.percentile(np.array([7]), 50)

assert p == 7
