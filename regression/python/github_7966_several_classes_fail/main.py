# Every object in a list carries the same type tag, so with several classes
# the element's class is unknown and len() is refused (#7966).
class C:

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n


class D:

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n + 100


i: int = 1
xs = [C(2), D(3)]
assert len(xs[i]) == 3
