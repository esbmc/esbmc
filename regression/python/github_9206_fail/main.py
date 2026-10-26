class A:

    def m(self) -> int:
        return 1


a = A()
if a.m() == 1:
    f = a.m
else:

    def f() -> int:
        return 2


assert f() == 2
