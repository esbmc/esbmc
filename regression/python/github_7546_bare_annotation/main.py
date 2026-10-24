# A bare annotation binds nothing, so B.m is still the inherited A.m (#7546).
class A:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class B(A):
    m: int


def main() -> None:
    assert B.m(10, 4) == 6


main()
