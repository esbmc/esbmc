def g(x: int) -> int:
    return x + 1

class B:
    def __init__(self) -> None:
        self.child = g

b = B()
assert b.child(1, 2) == 2
