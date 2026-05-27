import threading


# Two subclass-instance threads write the same module-level global
# without synchronisation. Under --data-races-check the spawn happens-
# before edge is not enough — ESBMC must report a W/W data race on
# `shared`.
shared: int = 0


class WriterA(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        global shared
        shared = 1


class WriterB(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        global shared
        shared = 2


a: WriterA = WriterA()
b: WriterB = WriterB()
a.start()
b.start()
a.join()
b.join()
