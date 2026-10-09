# Issue #8256: a nested class whose name the class body binds again is left in
# place, so the later binding is what Outer.Inner reads.
class A:

    class B:
        pass

    B: int = 3


class M:

    class B:

        @staticmethod
        def g() -> int:
            return 1

    @staticmethod
    def B() -> int:
        return 7


class W:

    class B:
        pass

    K: int = (B := 4) + B


assert A.B == 3
assert M.B() == 7
assert W.K == 8
