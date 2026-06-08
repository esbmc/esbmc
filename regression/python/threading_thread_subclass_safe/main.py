import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        global shared
        shared = 42


shared: int = 0
w: Worker = Worker()
w.start()
w.join()

# Read after join: the join happens-after edge makes the spawned
# thread's write to ``shared`` visible.
assert shared == 42
