# Issue #8357: called through the class, Outer.make runs on an Other, whose
# Inner differs, so `self.Inner` is left as written.
from helper import Other


class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def make(self, b: int) -> int:
        return self.Inner(b).b


assert Outer.make(Other(), 4) == 5
