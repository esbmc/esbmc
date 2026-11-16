class C:
    def m1(self) -> int:
        return 1


def two() -> int:
    return 2


o = C()
a = o.m1


def k() -> int:
    a = two
    return a()


assert a() == 1
assert k() == 2
