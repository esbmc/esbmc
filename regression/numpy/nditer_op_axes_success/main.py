import numpy as np

x = np.array([1, 2])
y = np.array([10, 20, 30])
total = 0
count = 0
for p, q in np.nditer([x, y], op_axes=[[0, -1], [-1, 0]]):
    total += p * q
    count += 1
assert count == 6
assert total == 180
