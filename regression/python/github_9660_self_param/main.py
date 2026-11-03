class A:
    def f(self, x: int, y: int, z: int = 1):
        return x + y - z


class B(A):
    def f(self, x: int, y: int, z: int = 1, m: int = 3, n: int = 5):
        return x + y - z + 10


def call(self):
    return self.f(1, 2, 3)


assert call(A()) == 0
assert call(B()) == 10
