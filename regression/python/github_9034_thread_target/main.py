import threading


def caller() -> None:
    t = threading.Thread(target=later, args=(1,))
    t.start()
    t.join()


def later(x: int) -> None:
    pass


caller()
