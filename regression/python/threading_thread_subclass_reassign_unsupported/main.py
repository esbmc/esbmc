import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        pass


w: Worker = Worker()
# Reassigning the subclass-bound variable would make the trampoline's
# read of ``w`` ambiguous; the validator refuses it.
w = Worker()
w.start()
