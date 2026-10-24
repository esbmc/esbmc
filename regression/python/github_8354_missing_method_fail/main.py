class C:
    def m1(self) -> int:
        return 1


o = C()
a, n = o.m2, 0
if n:
    a()
