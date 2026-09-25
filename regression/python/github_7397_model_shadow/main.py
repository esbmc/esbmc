# A class tag is keyed by name alone, so a user class shadowing an imported
# operational model's class was discarded and every use bound to the model's,
# reporting FAILED on an assertion CPython holds (#7397). Only a class built
# from a model file is exempt; an always-loaded model carries the program file
# as its module and is exempted by the same-file test instead.
import threading


class Thread:
    def __init__(self) -> None:
        self.n: int = 7

    def total(self) -> int:
        return self.n


t = Thread()
assert t.total() == 7
