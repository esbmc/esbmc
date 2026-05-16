import threading


def worker() -> None:
    pass


def reassign() -> None:
    t = threading.Thread(target=worker)
    t = threading.Thread(target=worker)
    t.start()
    t.join()


reassign()
