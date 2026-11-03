# D.m resolves to A.m (a - b), which is not converted yet when helper's body
# is. The call must not bind R.m, a later base that already has a symbol, and
# prove 10 + 4 (#7546).
class R:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a + b


class A:
    @staticmethod
    def helper() -> int:
        return D.m(10, 4)

    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class L(A):
    pass


class D(L, R):
    pass


def main() -> None:
    assert A.helper() == 14


main()
