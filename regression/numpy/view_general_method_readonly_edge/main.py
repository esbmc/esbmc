import numpy as np

a = np.array([[1, 2], [3, 4]])
d = a.diagonal()

class Consumer:
    def read(self, w):
        return w[0] + w[1]

c = Consumer()
assert c.read(d) == 5
