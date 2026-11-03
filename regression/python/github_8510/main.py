def size(board):
    return board.size + Board.base


def grow(board: "Board"):
    board.size = board.size + 1


class Board:
    base = 10

    def __init__(self):
        self.size = 2


b = Board()
assert size(b) == 12
grow(b)
assert size(b) == 13
assert Board.base == 10
