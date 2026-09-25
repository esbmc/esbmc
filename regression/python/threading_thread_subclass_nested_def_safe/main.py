import threading

total: int = 0


class Adder(threading.Thread):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount: int = amount

    def run(self) -> None:
        global total
        total = total + self.amount


def outer() -> None:
    def bump(a: int) -> int:
        return a + 1

    a: Adder = Adder(1)
    a.start()
    a.join()
    assert bump(1) == 2
    assert total == 1


outer()
