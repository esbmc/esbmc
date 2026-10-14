import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        x: int = 100
        # Assertion lives inside the spawned thread; symex must reach it.
        assert x < 50


w: Worker = Worker()
w.start()
w.join()
