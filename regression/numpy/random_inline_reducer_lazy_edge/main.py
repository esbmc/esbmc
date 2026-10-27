import numpy as np

x = np.random.random(2)
ok = len(x) == 3 or np.sum(np.random.random(3)) < 3.0
count = 0
while np.sum(np.random.random(2)) < 0.5 and count < 2:
    count += 1

assert ok
assert count <= 2
