class C:
    def m1(self) -> int:
        return 1


o = C()
a = o.m1


def a() -> int:
    return 2


assert a() == 1
