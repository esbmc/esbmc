# Issue #8963: the nested class's constructor runs, so the wrong value is caught.
class Op:

    def __init__(self, b: int):
        self.nested = self.Nested(b)

    class Nested:

        def __init__(self, b: int):
            self.b = b


op = Op(1)
assert type(op.nested.b) is int
assert op.nested.b == 2
