import numpy as np


class Counter:

    def __init__(self) -> None:
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return 1


c = Counter()
raised = False
try:
    np.full(3, c.next(), shape=2)
except TypeError:
    raised = True
assert raised
assert c.n == 1
