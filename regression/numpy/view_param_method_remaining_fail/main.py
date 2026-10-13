import numpy as np

class Box:
    def put(self, row):
        row[0] = 7
        return len(row)

a = np.array([1, 2, 3])
v = a[0:]
b = Box()
n = b.put(v)
assert a[0] == 7
