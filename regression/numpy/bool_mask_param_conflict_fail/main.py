import numpy as np

# conflicting call-site shapes for the same parameter
# are rejected rather than silently keeping the first-found shape.
def select(a, mask):
    return a[mask]

a2 = np.array([[1, 2], [3, 4]])
mask2 = np.array([True, False])
b = select(a2, mask2)

a3 = np.array([[1, 2], [3, 4], [5, 6]])
mask3 = np.array([True, False, True])
c = select(a3, mask3)
