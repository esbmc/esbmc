# Issue #8357: an instance attribute named Inner hides the nested class, so
# `self.Inner` is left as written.
class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def __init__(self, b: int):
        self.Inner = 3
        self.inner = self.Inner + b


o = Outer(4)
assert o.inner == 7
