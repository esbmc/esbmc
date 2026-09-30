# f(L) calls L.m, which is A.m (10 - 4), not D.m through the MRO of the
# class D (#7546).
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
    assert f(L) == 14


main()
