# Issue #8357: the nested class's constructor runs, so the wrong value is caught.
class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def __init__(self, b: int):
        self.inner = self.Inner(b)


o = Outer(4)
assert o.inner.b == 5
