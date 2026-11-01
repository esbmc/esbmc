class A:
    def f(self, x: int, y: int, z: int = 1):
        return x + y - z


class B(A):
    def f(self, x: int, y: int, z: int = 1, m: int = 3, n: int = 5):
        return x + y - z + 10


def f(cls):
    return cls.f(1, 2, 3)


assert f(A()) == 0
assert f(B()) == 10
