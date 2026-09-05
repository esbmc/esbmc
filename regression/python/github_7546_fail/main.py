# add(10, 4) is 6: binding the receiver as the first argument would shift both
# operands and give something else entirely (#7546).
class C:
    @staticmethod
    def add(a: int, b: int) -> int:
        return a - b


def main() -> None:
    d = C()
    assert d.add(10, 4) == 14


main()
