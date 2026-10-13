def size(board):
    return board.size


class Board:
    def __init__(self):
        self.size = 2


assert size(Board()) == 3
