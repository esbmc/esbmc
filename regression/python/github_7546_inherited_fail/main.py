# 10 - 4 is 6; 14 is what `a + b` would give. Pins the value the inherited
# @staticmethod actually computes, not merely that a verdict is reached (#7546).
class B:
    @staticmethod
    def add(a: int, b: int) -> int:
        return a - b


class D(B):
    pass


def main() -> None:
    d = D()
    assert d.add(10, 4) == 14


main()
