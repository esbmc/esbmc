# A @staticmethod has no implicit first argument, so calling one through an
# instance must not bind the receiver: both spellings of the call agree, and a
# @classmethod still takes its implicit cls (#7546).
class C:
    @staticmethod
    def author(name: int) -> int:
        return name

    @staticmethod
    def add(a: int, b: int) -> int:
        return a - b

    @classmethod
    def make(cls, name: int) -> int:
        return name

    def method(self, name: int) -> int:
        return name


def main() -> None:
    d = C()
    assert d.author(3) == 3
    assert C.author(3) == 3
    assert d.add(10, 4) == 6
    assert C.make(3) == 3
    assert d.method(3) == 3


main()
