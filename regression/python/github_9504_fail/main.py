class C:
    def n(self):
        return 5

    def f(self, a):
        if a == 0:
            return "s"
        else:
            return self.n()

c = C()
d = c.f(1)
assert d == 6
