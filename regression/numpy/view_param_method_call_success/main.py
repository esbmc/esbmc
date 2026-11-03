import numpy as np

class Processor:
    def run(self, row):
        return row[0]

a = np.array([5, 6, 7])
v = a[0:]
p = Processor()
result = p.run(v)
assert result == 5
