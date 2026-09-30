# D.m is C.m (10 + 4) by the MRO D, B, E, C, X, so helper does not return 6
# (#7546).
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
    assert X.helper() == 6


main()
