# Issue #9472: a method reading an attribute chain through a back-reference
class Square:
    def __init__(self, board):
        self.board = board
    def move(self):
        self.board.zobrist.add()
class Board:
    def __init__(self):
        self.sq = Square(self)
b = Board()
