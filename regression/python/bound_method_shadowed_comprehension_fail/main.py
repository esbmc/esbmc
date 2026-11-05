class C:
    def m1(self) -> int:
        return 1


def two() -> int:
    return 2


o = C()
a = o.m1
b = [a() for a in [two]]
assert b[0] == 1
