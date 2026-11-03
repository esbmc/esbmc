class M:
    def __init__(self, n):
        self.n = n

    def f(self):
        return self.n


a = M(1)
fs = [a.f]
a = M(2)
assert fs[0]() == 1
