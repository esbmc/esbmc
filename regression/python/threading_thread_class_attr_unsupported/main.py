import threading


def worker() -> None:
    pass


class Holder:
    t = threading.Thread(target=worker)
