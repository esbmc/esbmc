import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        pass

    # User override of start() would be silently bypassed by the
    # lowering (which rewrites `.start()` calls to spawn intrinsics).
    def start(self) -> None:
        pass


w: Worker = Worker()
w.start()
