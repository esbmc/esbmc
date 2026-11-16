class A:

    def m(self) -> int:
        return 1


a = A()
if a.m() == 1:
    f = a.m
else:

    class f:
        pass


assert f() == 1
