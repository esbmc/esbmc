import threading


class Counter:
    def __init__(self) -> None:
        self.value: int = 0


# Mixed args tuple: (int literal, class instance). Each tuple element
# drives its own typed prelude declaration independently — int args
# stay int-typed (= 0), instance args take the object-typed fallback
# (`: object = None`).
def worker(n: int, c: Counter) -> None:
    c.value = n


counter = Counter()
t = threading.Thread(target=worker, args=(7, counter))
t.start()
t.join()

assert counter.value == 7
