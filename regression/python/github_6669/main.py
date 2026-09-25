class Adder:
    def __init__(self, amount: int) -> None:
        self.amount: int = amount


obj: Adder = None


def reader() -> None:
    global obj
    assert obj.amount == 1


def outer() -> None:
    global obj

    # Converting this nested def used to clear the enclosing scope's `global`
    # declarations, so the assignment below bound a local instead.
    def bump(a: int) -> int:
        return a + 1

    obj = Adder(1)
    assert bump(1) == 2
    reader()


outer()
