class Op:

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.result = None

    def compute(self):
        self.result = self.a + self.b
        return self.result


def run(c: bool) -> None:
    op = Op(5, 10)
    other = Op("Hello", "World")
    setattr(op, "a", 6)
    op.compute()
    if c:
        other.compute()
    half = op.result / 2
    assert half == 8.0


run(True)
run(False)
