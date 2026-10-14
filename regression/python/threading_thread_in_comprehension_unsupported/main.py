import threading


def worker() -> None:
    pass


threads = [threading.Thread(target=worker) for _ in range(3)]
