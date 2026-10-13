# Issue #8357: getattr reaches Outer, whose make then runs on an Other, so
# `self.Inner` is left as written.
from helper import Other


class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def make(self, b: int) -> int:
        return self.Inner(b).b


x = Outer()
assert getattr(x, "__cl" + "ass__").make(Other(), 4) == 5
