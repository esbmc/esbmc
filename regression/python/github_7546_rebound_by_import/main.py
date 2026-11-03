# The import rebinds R before D is defined, so D derives from the imported class
# and D.m is A.m through L, not the earlier R's override (#7546).
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


from lib_7546_rebind_base import Q as R


class D(L, R):
    pass


def main() -> None:
    assert D.m(10, 4) == 6


main()
