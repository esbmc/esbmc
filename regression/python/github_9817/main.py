# 9817: an inherited method named like a module global
class D:
    def f(self):
        return 2

class E(D):
    pass

f = E().f()
assert f == 2
