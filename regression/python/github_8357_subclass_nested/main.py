# Issue #8357: a subclass nests its own Inner, so for a Sub instance
# `self.Inner` in Outer.make is Sub.Inner and is left as written.
class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def make(self, b: int) -> int:
        return self.Inner(b).b


class Sub(Outer):

    class Inner:

        def __init__(self, b: int):
            self.b = b + 1


assert Sub().make(4) == 5
