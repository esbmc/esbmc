import numpy as np

# An empty constructor array keeps the existing empty-sequence diagnostic.
a = np.zeros((0,))
b = np.mean(a)

assert b == 0
