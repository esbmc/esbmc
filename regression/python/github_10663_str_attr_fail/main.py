class Op:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.result = ""

    def compute(self):
        self.result = self.a + self.b
        return self.result


s = Op("a", "b")
q = s.compute()
assert q != "ab"
