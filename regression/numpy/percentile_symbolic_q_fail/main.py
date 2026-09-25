import numpy as np

q = int(input())
p = np.percentile(np.array([1, 2, 3]), q)

assert p >= 1
