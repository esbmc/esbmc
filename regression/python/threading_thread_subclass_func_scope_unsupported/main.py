import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        pass


def make_and_run() -> None:
    # Function-scope subclass binding is refused: the trampoline reads
    # the module-level variable; a function-local binding would not be
    # visible to the spawned thread.
    w: Worker = Worker()
    w.start()
    w.join()


make_and_run()
