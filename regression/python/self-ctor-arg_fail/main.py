class Board:
    def __init__(self):
        self.n = 3
        self.sq = Square(self)


class Square:
    def __init__(self, board):
        self.board = board


b = Board()
assert b.sq.board.n == 4
