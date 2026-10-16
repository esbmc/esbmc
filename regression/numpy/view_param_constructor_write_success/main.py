import numpy as np

class Box:
    def __init__(self, row):
        self.first = row[0]
        row[0] = 7

a = np.array([1, 2, 3])
v = a[0:]
b = Box(v)
assert a[0] == 7
