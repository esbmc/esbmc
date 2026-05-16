import threading


def worker() -> None:
    pass


for _ in range(3):
    t = threading.Thread(target=worker)
    t.start()
    t.join()
