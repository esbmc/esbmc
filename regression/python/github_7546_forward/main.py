# Leaf.m resolves to A.m, which is converted before helper's body is, so the
# inherited declaration binds (#7546).
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
    assert Mid.helper() == 6


main()
