class A:
    def f(self, x, y, z=1):
        return x + y - z


class B(A):
    def f(self, x, y, z=1, m=3, n=5):
        return x + y - z + 10


def f(cls):
    return cls.f(1, 2, 3)


assert f(A()) == 0
assert f(B()) == 10
