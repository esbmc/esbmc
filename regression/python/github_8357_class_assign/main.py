# Issue #8357: assigning __class__ changes what `self.Inner` finds, so it is
# left as written.
from helper import Other


class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def make(self, b: int) -> int:
        return self.Inner(b).b


o = Outer()
make = o.make
o.__class__ = Other
assert make(4) == 5
