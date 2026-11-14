class Hex:
    def __init__(self):
        self.ids = [10, 20]

    def get_by_id(self, i):
        return self.ids[i]


class Pos:
    def __init__(self, hex):
        self.hex = hex


class Done:
    def next_max(self, pos):
        return pos.hex.get_by_id(1)

    def next_cell(self, pos):
        return self.next_max(pos)


def find_moves(pos: Pos):
    done = Done()
    return done.next_cell(pos)


assert find_moves(Pos(Hex())) == 20
