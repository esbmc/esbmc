# KNOWNBUG (#8915): the receiver of first() has no known class, so the call is
# refused. Lowering it to a null constant read as 0, which matched the expected
# value only by chance: with range(5, 7) the SUCCESSFUL verdict still held.
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
assert b.squares[1].first() == 1
