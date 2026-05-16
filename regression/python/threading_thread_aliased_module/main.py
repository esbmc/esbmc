import threading as t


def worker() -> None:
    pass


x = t.Thread(target=worker)
x.start()
x.join()
