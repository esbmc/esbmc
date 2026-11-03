# Issue #8357: with multiple inheritance Other precedes Outer, so `self.Inner`
# in Outer.make is Other.Inner and is left as written.
from helper import Other


class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def make(self, b: int) -> int:
        return self.Inner(b).b


class Both(Other, Outer):
    pass


assert Both().make(4) == 5
