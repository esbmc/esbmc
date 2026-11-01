class A:
    def f(self, x, y, z=1):
        return x + y - z


class B(A):
    def f(self, x, y, z=1, m=3, n=5):
        return x + y - z + 10


def call(obj):
    return obj.f(1, 2, 3)


z1 = call(A())
z2 = call(B())
assert isinstance(z1, int)
assert isinstance(z2, int)
assert z1 == 0
assert z2 == 10
