class Op:
    def __init__(self, b):
        self.nested = self.Nested(b)

    class Nested:
        def __init__(self, b):
            self.b = b


op = Op(True)
assert type(op.nested.b) is int
