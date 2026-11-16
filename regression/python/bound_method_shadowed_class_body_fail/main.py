class C:
    def m1(self) -> int:
        return 1


def a() -> int:
    return 2


o = C()


class K:
    a = o.m1

    def f(self) -> int:
        return a()


assert K().f() == 1
