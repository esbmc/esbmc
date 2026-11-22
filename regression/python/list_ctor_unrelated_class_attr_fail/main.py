class Square:
    def __init__(self, n):
        self.n = n

    def move(self):
        return self.n


class Other:
    def __init__(self):
        self.k = 1

    def move(self):
        return 7



class Board:
    def __init__(self):
        self.squares = [Square(1)]


b = Board()
b.squares[0] = Other()
s = b.squares[0]
assert s.move() == 1
