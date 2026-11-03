# Issue #8256: a class nested in a class body is reachable as Outer.Inner, and
# by its bare name in the outer class body; a deeper nesting and a
# module-level class of the same name stay distinct.
class A:

    class B:

        @staticmethod
        def g(x: int) -> int:
            return x + 1

        class C:

            def __init__(self, v: int) -> None:
                self.v: int = v

    K: int = B.g(4)

    def read(self, c: B.C) -> int:
        return c.v


class B:

    @staticmethod
    def g(x: int) -> int:
        return x - 1


assert A.B.g(1) == 2
assert A.K == 5
assert A().read(A.B.C(6)) == 6
assert A.B.C(7).v == 7
assert B.g(1) == 0
c: A.B.C = A.B.C(3)
assert isinstance(c, A.B.C)
