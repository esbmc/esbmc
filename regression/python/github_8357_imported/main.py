# Issue #8357: Outer lives in another module, which cannot see that Sub binds
# Inner, so `self.Inner` there is left as written.
from helper import Outer


class Other:

    def __init__(self, b: int):
        self.b = b + 1


class Sub(Outer):
    Inner = Other


assert Sub().make(4) == 5
