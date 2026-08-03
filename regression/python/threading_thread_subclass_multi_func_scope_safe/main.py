import threading

a: int = 0
b: int = 0


class W1(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        global a
        a = 1


class W2(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        global b
        b = 2


def f1() -> None:
    w: W1 = W1()
    w.start()
    w.join()


def f2() -> None:
    w: W2 = W2()
    w.start()
    w.join()


f1()
f2()
assert a == 1
assert b == 2
