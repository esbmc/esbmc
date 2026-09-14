# D redeclares add as an instance method, so the base's @staticmethod does not
# apply and the receiver is bound again: the lookup has to stop at the first
# class that declares the method (#7546).
class B:
    @staticmethod
    def add(a: int, b: int) -> int:
        return a - b


class D(B):
    def add(self, a: int, b: int) -> int:
        return a + b


def main() -> None:
    d = D()
    assert d.add(10, 4) == 14


main()
