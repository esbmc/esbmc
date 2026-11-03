# D.m is R.m (10 + 4) by the MRO D, L, R, A, so it does not return A.m's 6
# (#7546).
class A:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class L(A):
    pass


class R(A):
    @staticmethod
    def m(a: int, b: int) -> int:
        return a + b


class D(L, R):
    pass


def main() -> None:
    assert D.m(10, 4) == 6


main()
