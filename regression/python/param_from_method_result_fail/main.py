class Hex:
    def __init__(self):
        self.ids = [10, 20]

    def get_by_id(self, i: int) -> int:
        return self.ids[i]


class Pos:
    def __init__(self, hex):
        self.hex = hex

    def clone(self):
        return Pos(self.hex)


def first(pos):
    return pos.hex.get_by_id(1)


assert first(Pos(Hex()).clone()) == 10
