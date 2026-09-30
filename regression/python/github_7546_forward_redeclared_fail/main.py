# Leaf.m is Leaf's own m (a + b), not converted yet when helper's body is. The
# call must not bind the inherited A.m and prove 10 - 4 (#7546).
class A:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class Leaf(A):
    @staticmethod
    def helper() -> int:
        return Leaf.m(10, 4)

    @staticmethod
    def m(a: int, b: int) -> int:
        return a + b


def main() -> None:
    assert Leaf.helper() == 6


main()
