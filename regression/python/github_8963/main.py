# Issue #8963: a type() call compared only with `is` cannot reach a method, so
# `self.Nested` still finds the class nested in the method's class.
class Op:

    def __init__(self, b: int):
        self.nested = self.Nested(b)

    class Nested:

        def __init__(self, b: int):
            self.b = b


op = Op(1)
x = 1
assert type(x) is int
assert type(op.nested.b) is int
assert op.nested.b == 1
