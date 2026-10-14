import threading


class Worker(threading.Thread):
    def __init__(self) -> None:
        super().__init__()

    def run(self) -> None:
        pass


flag: bool = True
if flag:
    w: Worker = Worker()
    w.start()
    w.join()
