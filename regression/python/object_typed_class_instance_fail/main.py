class Box:
    def __init__(self, v: int) -> None:
        self.v = v


t: object = Box(7)
assert t.v == 0
