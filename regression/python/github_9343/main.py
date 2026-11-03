class C:

    @staticmethod
    def f(x):
        return x


r = C.f(1)
assert r == 1
