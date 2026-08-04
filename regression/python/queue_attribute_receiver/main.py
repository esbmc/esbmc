import queue


class Holder:
    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()

    def add(self, v: int) -> None:
        self.q.put(v)

    def take(self) -> int:
        return self.q.get()


h: Holder = Holder()
h.add(5)
assert h.take() == 5
