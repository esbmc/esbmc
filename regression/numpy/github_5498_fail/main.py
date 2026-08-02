# Negative variant of github_5498: a % 3 is [1, 2, 0], so asserting b[0] == 2
# must fail.
import numpy as np

a = np.array([4, 5, 6])
b = a % 3
assert b[0] == 2
