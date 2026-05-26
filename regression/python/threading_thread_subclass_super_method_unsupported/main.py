import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        # super().run() is a non-init super call; we cannot soundly
        # strip it the way we strip super().__init__(), so the
        # validator refuses it rather than silently model object.run().
        super().run()


w: Worker = Worker()
w.start()
