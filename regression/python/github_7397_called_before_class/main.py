# g() runs before the local class, so it reads the imported X; the collision
# stays refused (#7397).
from m import X


def g() -> int:
    return X().v


r: int = g()


class X:
    def __init__(self) -> None:
        self.v: int = 2


assert r == 1
