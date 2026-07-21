import numpy as np

# A reassigned mask now goes through the symbolic path, which
# reads the mask's live runtime value directly (not its first AST
# declaration), so reassignment before use is sound.
a = np.array([[1, 2], [3, 4], [5, 6]])
mask = np.array([True, False, False])
mask = np.array([False, False, True])
b = a[mask]
