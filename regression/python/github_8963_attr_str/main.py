class Op:
    def __init__(self, b):
        self.nested = self.Nested(b)

    class Nested:
        def __init__(self, b):
            self.b = b


op = Op("s")
assert type(op.nested.b) is str
assert type(op.nested.b) is not int
