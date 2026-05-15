class T:
    def __init__(self) -> None:
        self.acc: int = 0
    def __setitem__(self, key, value: int) -> None:
        sl0: slice = key[0]
        self.acc = value + sl0.stop - sl0.start

t: T = T()
t[2:5, 3:8] = 100
assert t.acc == 103
