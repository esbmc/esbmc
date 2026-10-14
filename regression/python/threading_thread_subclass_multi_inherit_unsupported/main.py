import threading


class Mixin:
    pass


class Worker(threading.Thread, Mixin):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        pass


w: Worker = Worker()
