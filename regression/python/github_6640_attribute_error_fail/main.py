class Holder:
    def __init__(self, v: int) -> None:
        self.v = v


h = Holder(1)
assert h.missing(1) == 2
