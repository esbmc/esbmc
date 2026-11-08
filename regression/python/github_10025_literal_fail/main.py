class Board:
    def __init__(self, empty: bool):
        self.cells = None if empty else [[0, 1], [2]]

    def drop(self, i: int, v: int) -> None:
        self.cells[i].remove(v)


b = Board(False)
b.drop(0, 0)
assert b.cells[0] == [1]
assert b.cells[0] == [0, 1]
