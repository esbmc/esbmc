class C:
    def m1(self) -> int:
        return 1

    def m2(self) -> int:
        return 2


o = C()
a, b = o.m1, o.m2
assert a() == 2
assert b() == 2
