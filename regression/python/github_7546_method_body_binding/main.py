# The local m in helper's body binds in that function's scope, so B does not
# hide A.m and B.m is the inherited one (#7546).
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
    assert B.m(10, 4) == 6


main()
