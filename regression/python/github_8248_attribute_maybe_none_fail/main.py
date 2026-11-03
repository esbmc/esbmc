class R:
    def __init__(self) -> None:
        self.r = None


def fill(o: R, flag: bool) -> None:
    if flag:
        o.r = 3


o = R()
fill(o, False)
y = o.r + 1
