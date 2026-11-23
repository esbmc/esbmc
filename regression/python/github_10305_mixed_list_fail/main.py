class A:

    def m(self, x: int) -> int:
        return x


class B(A):

    def m(self, x: int) -> int:
        return 10 // x


for o in [A(), B()]:
    o.m(0)
