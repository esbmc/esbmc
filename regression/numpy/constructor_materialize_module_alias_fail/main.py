import numpy as np


# A user class whose own method happens to share a name with a numpy
# constructor (full) must not be treated as if it were numpy's full() just
# because the attribute name matches -- the receiver is not the numpy
# module, so the call is not a numpy array and must be rejected explicitly
# rather than materialized from the shape/fill arguments.
class Grid:
    def full(self, shape, fill):
        return [fill, fill, fill, fill, fill]


g = Grid()
a = g.full((2, 2), 9)
b = np.sum(a)

assert b == 45
