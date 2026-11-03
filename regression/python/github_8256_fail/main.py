# Issue #8256: the nested class's method runs, so the wrong result is caught.
class A:

    class B:

        @staticmethod
        def g(x: int) -> int:
            return x + 1


assert A.B.g(1) == 1
