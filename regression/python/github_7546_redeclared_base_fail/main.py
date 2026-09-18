# D derives from the second B, whose m adds, so D.m does not return A.m's 6
# (#7546).
class A:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class B(A):
    pass


class B:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a + b


class D(B):
    pass


def main() -> None:
    assert D.m(10, 4) == 6


main()
