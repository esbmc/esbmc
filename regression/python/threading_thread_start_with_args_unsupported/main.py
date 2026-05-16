import threading


def worker() -> None:
    pass


t = threading.Thread(target=worker)
t.start()
t.join(1.0)
