class D:
    def f(self):
        return 2

class E(D):
    pass

f = E().f()
assert f == 3
