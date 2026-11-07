class Op:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def compute(self):
        return self.a + self.b


r1 = Op(1, 2).compute()
r2 = Op("x", "y").compute()
assert isinstance(r2, str)
assert r1 == 3
assert r2 == "xy"
