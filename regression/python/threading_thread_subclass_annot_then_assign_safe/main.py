import threading

total: int = 0


class Adder(threading.Thread):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount: int = amount

    def run(self) -> None:
        global total
        total = total + self.amount


def spawn() -> None:
    a: Adder
    a = Adder(5)
    a.start()
    a.join()
    assert total == 5


spawn()
