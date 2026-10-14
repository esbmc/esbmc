import threading


def worker() -> None:
    pass


config = {"target": worker}
t = threading.Thread(**config)
