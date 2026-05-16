import threading


def worker() -> None:
    pass


if True:
    t = threading.Thread(target=worker)
    t.start()
    t.join()
