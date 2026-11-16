class D:
    def __init__(self):
        self.p = [0, 0, 1]

    def f(self, i):
        return self.p[i]

    def same(self, i, j):
        return self.f(i) == self.f(j)


d = D()
assert d.same(0, 1)
assert d.same(0, 2)
