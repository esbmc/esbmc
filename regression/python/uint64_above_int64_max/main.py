class uint64(int):
    pass


class Box:
    def __init__(self, v: uint64):
        self.v = v


def half(n: uint64) -> uint64:
    x = n
    assert x >= 1
    return (x + 1) // 2


def ident(n: uint64) -> uint64:
    return n


big = uint64(2**63 + 5)
assert half(big) == 2**62 + 3
assert ident(big) >= 1
assert Box(big).v >= 1
