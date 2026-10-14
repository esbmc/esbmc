import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        pass


def build() -> None:
    # Same single-definition rule as at module scope: the trampoline's
    # read of the hoisted global would otherwise be ambiguous.
    w: Worker = Worker()
    w = Worker()
    w.start()


build()
