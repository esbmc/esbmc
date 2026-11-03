# Issue #8357: a subclass binds Inner, so for a Sub instance `self.Inner` in
# Outer.make is Other, not the nested class, and is left as written.
class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def make(self, b: int) -> int:
        return self.Inner(b).b


class Other:

    def __init__(self, b: int):
        self.b = b + 1


class Sub(Outer):
    Inner = Other


assert Sub().make(4) == 5
