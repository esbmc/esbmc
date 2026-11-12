class Grid:
    def __init__(self):
        self.cells = [[1, 2], [3, 4]]

    def __getitem__(self, key):
        return self.cells[key[0]][key[1]]


g = Grid()
assert g[1, 0] == 3
