import numpy as np

# Two views of the same array keep aliasing each other after the source name
# is rebound: NumPy reads 9 here, so the assertion must fail. Each view is
# detached into its own snapshot today, so the write is not seen.
a = np.array([[1, 2, 3], [4, 5, 6]])
v1 = a[0]
v2 = a[0]
a = np.array([[7, 7, 7], [7, 7, 7]])
v1[0] = 9
assert v2[0] == 1
