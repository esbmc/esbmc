class Square:
    def __init__(self, board, pos):
        self.board = board
        self.pos = pos

    def first(self):
        return self.board.squares[0].pos


class Board:
    def __init__(self):
        self.squares = [Square(self, p) for p in range(2)]


b = Board()
assert b.squares[1].first() == 0
