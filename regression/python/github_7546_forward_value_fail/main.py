# Leaf.m binds the inherited A.m, so helper returns 10 - 4, not 14 (#7546).
class A:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class Mid(A):
    @staticmethod
    def helper() -> int:
        return Leaf.m(10, 4)


class Leaf(Mid):
    pass


def main() -> None:
    assert Mid.helper() == 14


main()
