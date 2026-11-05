class A:
    def __init__(self, a):
        self.a = a

    def compute(self):
        self.r = self.a
        return self.r


r = A(1).compute()
assert isinstance(r, int)
