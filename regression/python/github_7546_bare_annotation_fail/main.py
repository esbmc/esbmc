# B.m is the inherited A.m (10 - 4), so it does not return 14 (#7546).
class A:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class B(A):
    m: int


def main() -> None:
    assert B.m(10, 4) == 14


main()
