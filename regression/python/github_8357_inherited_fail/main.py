# Issue #8357: the base class's nested class runs, so the wrong value is caught.
# Derived.unit has no parameter, so there is no receiver to rewrite.
class Base:

    class Inner:

        def __init__(self, b: int):
            self.b = b


class Derived(Base):

    def __init__(self, b: int):
        self.inner = self.Inner(b)

    def unit() -> int:
        return 1


d = Derived(4)
assert d.inner.b == 5
