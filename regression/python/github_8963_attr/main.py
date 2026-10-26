class Op:
    def __init__(self, b):
        self.nested = self.Nested(b)

    class Nested:
        def __init__(self, b):
            self.b = b


op = Op(1)
assert type(op.nested.b) is int
assert type(op.nested.b) is not float
