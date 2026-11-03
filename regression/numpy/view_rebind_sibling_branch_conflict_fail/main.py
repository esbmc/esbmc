import numpy as np

c = nondet_bool()
a = np.array([1, 2, 3, 4])
if c:
    x = a[0:2]
else:
    x = a[1:4]
y = a[1:3]
a = np.array([9, 9, 9, 9])
x[0] = 7
assert y[0] == 2
