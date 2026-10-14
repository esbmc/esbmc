import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        pass


def build() -> None:
    for i in range(2):
        w: Worker = Worker()
        w.start()
        w.join()


build()
