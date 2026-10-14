import threading


class Worker(threading.Thread):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.seed: int = seed

    def run(self) -> None:
        global out
        out = self.seed * 2


out: int = 0
w: Worker = Worker(7)
w.start()
w.join()

assert out == 14
