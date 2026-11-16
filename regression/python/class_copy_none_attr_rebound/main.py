class Op:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.result = None

    def compute(self):
        self.result = self.a + self.b
        return self.result


x = 1
y = 2
x = 3
n = Op(x, y)
r = n.compute()
assert isinstance(r, int)
