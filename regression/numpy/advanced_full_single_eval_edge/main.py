import numpy as np


class Counter:

    def __init__(self) -> None:
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return 7


c = Counter()
a = np.full(3, c.next())
assert c.n == 1
assert a[0] == 7 and a[2] == 7

narrow = np.full(2, c.next() + 0.5, dtype=np.int16)
assert c.n == 2
assert narrow[1] == 7 and narrow.itemsize == 2
