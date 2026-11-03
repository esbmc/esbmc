# Issue #8357: a static method's first parameter is not the instance, so
# `self.Inner` is left as written.
from helper import Other


class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    @staticmethod
    def make(self, b: int) -> int:
        return self.Inner(b).b


assert Outer().make(Other(), 4) == 5
