# The MRO of D is D, B, E, C, X, so D.m is C.m even though X.m, which is not
# converted yet, is reached first depth-first and breadth-first (#7546).
class C:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a + b


class X:
    @staticmethod
    def helper() -> int:
        return D.m(10, 4)

    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class B(X):
    pass


class E(C, X):
    pass


class D(B, E):
    pass


def main() -> None:
    assert X.helper() == 14


main()
