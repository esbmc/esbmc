# Issue #8256: `A.B` is rewritten by name, so an outer class whose name is also
# a parameter is not hoisted; here `A` is a Q, whose B.g returns x + 1.
class Q:

    class B:

        @staticmethod
        def g(x: int) -> int:
            return x + 1


class A:

    class B:

        @staticmethod
        def g(x: int) -> int:
            return x - 1


def f(A: Q) -> int:
    return A.B.g(1)


assert f(Q()) == 2
