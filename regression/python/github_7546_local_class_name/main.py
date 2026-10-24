# f's parameter D is not the class D, so the call resolves against whatever
# class is passed, and the MRO of D must not decide it (#7546).
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


def f(D) -> int:
    return D.m(10, 4)


def main() -> None:
    assert f(L) == 6


main()
