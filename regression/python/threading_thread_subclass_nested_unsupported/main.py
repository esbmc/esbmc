import threading


def make_class():
    # Nested subclass def is refused: the module-top collector would
    # not see it, and the unstripped Thread base would crash the
    # converter at the call site.
    class Worker(threading.Thread):
        def __init__(self) -> None:
            super().__init__()

        def run(self) -> None:
            pass

    return Worker


W = make_class()
w = W()
