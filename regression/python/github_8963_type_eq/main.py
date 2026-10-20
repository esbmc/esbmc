# Issue #8963: `==` may hand the class from type() to a user `__eq__`, which
# could call a method on another class's object, so `self.Nested` is left as
# written.
class Op:

    def __init__(self, b: int):
        self.nested = self.Nested(b)

    class Nested:

        def __init__(self, b: int):
            self.b = b


op = Op(1)
assert type(op.nested.b) == int
