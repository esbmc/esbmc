import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()


w = Worker()
