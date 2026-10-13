import numpy as np

class Wrapper:
    def __init__(self, data):
        self.val = data[0]

a = np.array([3, 4, 5])
v = a[0:]
w = Wrapper(v)
assert w.val == 3
