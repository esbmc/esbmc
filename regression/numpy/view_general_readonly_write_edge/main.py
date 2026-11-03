import numpy as np

a = np.array([[1, 2], [3, 4]])
d = a.diagonal()

class Writer:
    def put(self, w):
        w[0] = 9
        return 0

r = Writer().put(d)
