# Issue #9472
class Square:
    def __init__(self, board):
        self.board = board
    def move(self) -> int:
        return self.board.zobrist.add(2)
class Hash:
    def __init__(self):
        self.v = 1
    def add(self, x: int) -> int:
        self.v = self.v + x
        return self.v
class Board:
    def __init__(self):
        self.zobrist = Hash()
        self.sq = Square(self)
b = Board()
r = b.sq.move()
assert r == 4
