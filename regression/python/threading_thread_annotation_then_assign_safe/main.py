import threading


def worker() -> None:
    pass


t: object
t = threading.Thread(target=worker)
