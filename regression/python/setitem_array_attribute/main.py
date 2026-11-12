from array import array


class Grid:
    def __init__(self):
        self.data = array("i", [1, 2, 3])


g = Grid()
g.data[2] = 7
assert g.data[2] == 7
