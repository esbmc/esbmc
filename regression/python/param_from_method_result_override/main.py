# A subclass overrides clone() to return another class, so the class of
# `p.clone()` is not known from Pos alone.
class Hex:
    def __init__(self):
        self.ids = [10, 20]

    def get_by_id(self, i: int) -> int:
        return self.ids[i]


class Hex2:
    def get_by_id(self, i: int) -> int:
        return 99


class Other:
    def __init__(self):
        self.hex = Hex2()


class Pos:
    def __init__(self, hex):
        self.hex = hex

    def clone(self):
        return Pos(self.hex)


class Sub(Pos):
    def clone(self):
        return Other()


def first(pos):
    return pos.hex.get_by_id(1)


def run(p: Pos):
    return first(p.clone())


assert run(Sub(Hex())) == 99
