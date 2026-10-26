import threading


def caller() -> None:
    t = threading.Thread(target=later, args=(1,))
    t.start()
    t.join()


_gap = 0


def later(x: int) -> None:
    pass


caller()
