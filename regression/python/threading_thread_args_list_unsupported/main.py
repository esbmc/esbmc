import threading


def worker(x: int) -> None:
    pass


payload = [1, 2, 3]
t = threading.Thread(target=worker, args=payload)
