import numpy as np

a = np.array([[1, 2], [3, 4]])
d = a.diagonal()

class Summary:
    def __init__(self, w):
        self.total = w[0] + w[1]

s = Summary(d)
assert s.total == 5
