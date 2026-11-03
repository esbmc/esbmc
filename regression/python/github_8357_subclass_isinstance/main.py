# Issue #8357: Outer is only constructed, subclassed and tested by isinstance,
# so `self.Inner` in Outer.__init__ is still the nested class for a Sub.
class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def __init__(self, b: int):
        self.inner = self.Inner(b)


class Sub(Outer):
    pass


o = Sub(4)
assert isinstance(o, Outer)
assert o.inner.b == 4
