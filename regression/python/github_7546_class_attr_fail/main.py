# B.m is the class attribute, which hides A.m: the call must not bind A.m and
# prove 10 - 4 (#7546).
class A:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a - b


class B(A):
    m = 5


def main() -> None:
    assert B.m(10, 4) == 6


main()
