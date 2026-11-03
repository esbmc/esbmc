class M:
    def f(self):
        return 1

    def f2(self):
        return 2


a = M()
c, d = (a.f, a.f2)
e = a.f2
assert c() == 2
