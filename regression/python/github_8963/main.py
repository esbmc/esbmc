# Issue #8963: a type() call that only feeds `is` must not stop `self.Nested`
# from resolving to the hoisted class.
class Op:

    def __init__(self, b):
        self.nested = self.Nested(b)

    class Nested:

        def __init__(self, b):
            self.b = b


op = Op(1)
x = 1
assert type(x) is int
assert type(x) is not float
assert op.nested.b == 1
