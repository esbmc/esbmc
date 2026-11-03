# Issue #8357: __getattribute__ decides what `self.Inner` is (Other, for any
# five-letter name), so it is left as written.
class Other:

    def __init__(self, b: int):
        self.b = b + 1


class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def make(self, b: int) -> int:
        return self.Inner(b).b

    def __getattribute__(self, name: str):
        return Other if len(name) == 5 else object.__getattribute__(self, name)


assert Outer().make(4) == 5
