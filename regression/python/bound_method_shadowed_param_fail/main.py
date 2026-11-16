class C:
    def m1(self) -> int:
        return 1


def two() -> int:
    return 2


o = C()
a = o.m1


def k(a) -> int:
    return a()


fs = [two]
assert k(fs[0]) == 1
