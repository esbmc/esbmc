import numpy as np

# An empty constructor array (0-sized shape) must keep the existing
# empty-sequence rejection for min(), not silently misread the shape as
# data or return an arbitrary value.
a = np.zeros((0,))
b = np.min(a)

assert b == 0
