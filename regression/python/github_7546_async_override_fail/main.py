# B.m is the async override, which is not converted to a function. The call
# must not bind A.m, the next declaration in the MRO, and prove 10 + 4 (#7546).
class A:
    @staticmethod
    def m(a: int, b: int) -> int:
        return a + b


class B(A):
    @staticmethod
    async def m(a: int, b: int) -> int:
        return a - b


def main() -> None:
    assert B.m(10, 4) == 14


main()
