class R:
    def __init__(self) -> None:
        self.r = None


def fill(o: R, flag: bool) -> None:
    if flag:
        o.r = 3


o = R()
fill(o, True)
assert o.r + 1 == 4
p = R()
fill(p, False)
assert p.r is None
