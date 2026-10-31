class C:
    def f(self, a):
        if a == 0:
            return "s"
        else:
            return 5

c = C()
d = c.f(1)
assert d == 5
