class C:

    @staticmethod
    def f(x):
        return x


r = C.f("ab")
assert len(r) == 2
