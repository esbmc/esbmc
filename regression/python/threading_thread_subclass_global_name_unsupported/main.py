import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        pass


w: Worker


def build() -> None:
    # The instance is a module global here, so the hoist that makes a
    # function-local binding visible to the trampoline does not apply.
    global w
    w = Worker()
    w.start()
    w.join()


build()
