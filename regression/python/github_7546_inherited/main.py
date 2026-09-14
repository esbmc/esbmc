# A @staticmethod inherited from a base binds no receiver either, so the
# derived-class call must not pass the instance as the first argument (#7546).
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
