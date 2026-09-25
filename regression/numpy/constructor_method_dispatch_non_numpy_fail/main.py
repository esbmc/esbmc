import numpy as np

# A plain class with no `min` method is not a numpy array; calling .min()
# on it must keep raising the ordinary Python AttributeError and must not
# be silently rewritten into a np.min(...) dispatch just because the
# method name overlaps with a numpy reducer.
class Bag:
    def __init__(self, value):
        self.value = value


not_numpy = Bag(5)
result = not_numpy.min()
assert result == 5
