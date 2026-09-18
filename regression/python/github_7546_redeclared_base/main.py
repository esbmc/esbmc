# B is declared twice; D derives from the second B, whose m adds. The first
# B's base A must not stand in for it (#7546).
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
    assert D.m(10, 4) == 14


main()
