import numpy as np

# A non-boolean mask is rejected generically at the subscript call site,
# before dispatch to bool-mask row selection - this already holds for the
# 2-D row-select path independently of literal vs. symbolic masks.
a = np.array([[1, 2], [3, 4]])
mask = np.array([1, 0])
b = a[mask]
