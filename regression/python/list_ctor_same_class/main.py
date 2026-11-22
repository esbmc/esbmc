class Square:
    def __init__(self, n):
        self.n = n

    def move(self):
        return self.n


sq = [Square(1)]
sq[0] = Square(4)
s = sq[0]
assert s.move() == 4
