import queue


class Holder:
    def __init__(self, q: queue.Queue) -> None:
        self.q: queue.Queue = q

    def take(self) -> int:
        return self.q.get()


shared: queue.Queue = queue.Queue()
shared.put(5)
h: Holder = Holder(shared)
assert h.take() == 5
