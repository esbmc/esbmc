class Op:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def compute(self):
        return self.a + self.b


r2 = Op("x", "y").compute()
assert r2 == "yx"
