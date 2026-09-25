import numpy as np

# sum() flattens through 1D/2D constructor literals only; a 3D constructor
# array must keep the existing explicit diagnostic instead of silently
# misreading the shape as data or crashing.
a = np.zeros((2, 2, 2))
b = np.sum(a)

assert b == 0
