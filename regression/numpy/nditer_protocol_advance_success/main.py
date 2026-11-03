import numpy as np

a = np.array([5, 6, 7])
it = np.nditer(a)
assert it.value == 5
x = next(it)
assert x == 5
assert it.iterindex == 0
y = next(it)
assert y == 6
assert it.iterindex == 1
it.iternext()
assert it.value == 7

total = 0
for v in iter(np.nditer(a)):
    total += int(v)
assert total == 18
