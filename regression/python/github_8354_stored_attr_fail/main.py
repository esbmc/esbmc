class C:
    def m1(self) -> int:
        return 1


def two() -> int:
    return 2


o = C()
a, n = o.m1, 0
o.m1 = two
assert a() == 2
