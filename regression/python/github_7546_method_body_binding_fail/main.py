# B.m is the inherited A.m (10 - 4), not 14; the local m in helper binds only
# in that function (#7546).
class A:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class B(A):
    @staticmethod
    def helper() -> int:
        m = 5
        return m


def main() -> None:
    assert B.helper() == 5
    assert B.m(10, 4) == 14


main()
