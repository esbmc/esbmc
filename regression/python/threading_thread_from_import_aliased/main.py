from threading import Thread as MyThread


def worker() -> None:
    pass


t = MyThread(target=worker)
