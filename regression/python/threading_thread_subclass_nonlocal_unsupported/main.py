import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        pass


def outer() -> None:
    w: Worker = Worker()

    def clobber() -> None:
        nonlocal w
        w = None

    w.start()
    w.join()


outer()
