class D:
    def f(self, x):
        return x

    def same(self, i):
        return self.f(i)


d = D()
assert d.f("a") == "a"
assert d.f(1) == 1
assert d.same(2) == 2
