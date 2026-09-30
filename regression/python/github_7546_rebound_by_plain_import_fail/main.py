# D's base R is the imported class, so D.m is A.m (10 - 4), not the earlier
# R's 14 (#7546).
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


from lib_7546_rebind_plain import R


class D(L, R):
    pass


def main() -> None:
    assert D.m(10, 4) == 14


main()
