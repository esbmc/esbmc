import numpy as np


class Counter:

    def __init__(self) -> None:
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return 3


c = Counter()
raised = False
try:
    np.linspace(0, 1, size=c.next())
except TypeError:
    raised = True
assert raised and c.n == 1
