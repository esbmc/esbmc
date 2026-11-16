class A:
    def f(self, x):
        return x + 1


def f(cls):
    return 100


a = A()
assert a.f(1) == 3
assert f(a) == 100
