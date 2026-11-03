# Issue #8357: the method rebinds self, so `self.Inner` is left as written.
from helper import Other


class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def make(self, b: int) -> int:
        self = Other()
        return self.Inner(b).b


assert Outer().make(4) == 5
