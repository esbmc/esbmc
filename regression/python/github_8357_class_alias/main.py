# Issue #8357: an alias of Outer calls make on an Other, so `self.Inner` is
# left as written.
from helper import Other


class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def make(self, b: int) -> int:
        return self.Inner(b).b


o = Outer
assert o.make(Other(), 4) == 5
