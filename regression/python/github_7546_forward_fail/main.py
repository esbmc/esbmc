# Leaf.m resolves to Mid.m (a + b), which is not converted yet when helper's
# body is. The call must bind neither A.m, the next declaration in the MRO,
# nor the module-level m, and prove 10 - 4 (#7546).
def m(a: int, b: int) -> int:
    return a - b


class A(object):
    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class Mid(A):
    @staticmethod
    def helper() -> int:
        return Leaf.m(10, 4)

    @staticmethod
    def m(a: int, b: int) -> int:
        return a + b


class Leaf(Mid):
    pass


def main() -> None:
    assert Mid.helper() == 6


main()
