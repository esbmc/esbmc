import threading


class Holder:
    def __init__(self) -> None:
        self.value: int = 0


def setter(h: Holder, x: int) -> None:
    h.value = x


h = Holder()
t = threading.Thread(target=setter, args=(h, 42))
t.start()
t.join()

assert h.value == 42
