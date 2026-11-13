import numpy as np

a = np.array([1.0, 2.0])
i = np.array([1, 2])
o = np.array([5.0, 5.0])
oi = np.array([5, 5])
caught = 0
try:
    np.add(a, a, out=np.zeros(3))
except ValueError:
    caught += 1
try:
    np.divide(i, i, out=oi)
except TypeError:
    caught += 1
try:
    np.multiply(a, a, where=np.array([1, 0]), out=o)
except TypeError:
    caught += 1
try:
    np.add(a, np.array([1.0, 2.0, 3.0]), out=o)
except ValueError:
    caught += 1
assert caught == 4
assert o[0] == 5.0 and o[1] == 5.0 and oi[0] == 5
