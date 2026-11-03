class C:
    def m1(self) -> int:
        return 1


def g() -> int:
    return 5


o = C()
a = o.m1
a, n = g, 0
assert a() == 5
