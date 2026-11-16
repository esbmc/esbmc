class A:
    def f(self, x, y, z=1):
        return x + y - z


def call(obj):
    return obj.f(1, 2, 3)


z = call(A())
assert isinstance(z, int)
assert z == 1
