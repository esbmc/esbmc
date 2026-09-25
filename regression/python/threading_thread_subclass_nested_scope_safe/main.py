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
    def inner() -> None:
        a: Adder = Adder(2)
        a.start()
        a.join()

    a: Adder = Adder(1)
    a.start()
    a.join()
    inner()
    assert total == 3


outer()
