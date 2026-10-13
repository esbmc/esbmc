def size(board: "Board") -> int:
    return board.size


class Board:
    def __init__(self, n: int):
        self.size = n


assert size(Board(2)) == 2
