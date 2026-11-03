class C:
    def m1(self) -> int:
        return 1


def two() -> int:
    return 2


o = C()
a, n = o.m1, 0


def k(a) -> int:
    return a()


assert k(two) == 1
