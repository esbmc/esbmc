class D:

    def __init__(self, x: int) -> None:
        self.x = x


d = D(1)
xs = [d]
assert d in xs
assert xs.count(d) == 1
assert xs.index(d) == 0
assert 5 in [1, 5]
