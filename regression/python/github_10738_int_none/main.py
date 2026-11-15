class Op:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.result = None

    def compute(self):
        self.result = self.a + self.b
        return self.result


op1 = Op(1, 2)
if nondet_bool():
    r1 = op1.compute()
else:
    r1 = None
assert op1.result == r1 or r1 is None
