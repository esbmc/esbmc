import threading


def build() -> None:
    w: Worker = Worker()
    w.start()
    w.join()


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        pass


build()
