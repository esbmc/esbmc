# Issue #8357: inside a method, `self.Inner` finds the class nested in the
# method's class, as `Outer.Inner` does.
class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def __init__(self, b: int):
        self.inner = self.Inner(b)


o = Outer(4)
assert o.inner.b == 4
