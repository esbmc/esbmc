def f(b: "Board") -> int:
    return b.x


class Base:
    def __init__(self):
        self.x = 5


class Board(Base):
    pass


assert f(Board()) == 6
