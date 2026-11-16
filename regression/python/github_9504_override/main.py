class C:
    def n(self):
        return 5

    def f(self, a):
        if a == 0:
            return "s"
        else:
            return self.n()


class D(C):
    def n(self):
        return "x"


d = D().f(1)
assert d == "x"
