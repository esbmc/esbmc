# Issue #9472
class Square:
    def __init__(self, board):
        self.board = board
    def move(self) -> int:
        return self.board.size()
class Board:
    def __init__(self):
        self.sq = Square(self)
    def size(self) -> int:
        return 8
b = Board()
assert b.sq.move() == 8
