# A @staticmethod inherited from a base is still classified as an instance
# method, so the receiver is bound and the operands shift. Pre-existing: the
# decorator lookup searches the receiver's own class body only (#7546).
class B:
    @staticmethod
    def add(a: int, b: int) -> int:
        return a - b


class D(B):
    pass


def main() -> None:
    d = D()
    assert d.add(10, 4) == 6


main()
