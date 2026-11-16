class A:
    def __init__(self, a):
        self.a = a

    def compute(self):
        self.r = self.a
        return self.r


r = A("ab").compute()
assert isinstance(r, str)
assert r == "ab"
