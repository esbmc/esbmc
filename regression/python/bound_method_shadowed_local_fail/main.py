class C:
    def m1(self) -> int:
        return 1


def two() -> int:
    return 2


o = C()
a = o.m1


def k(fs: list) -> int:
    for a in fs:
        return a()
    return 0


assert k([two]) == 1
