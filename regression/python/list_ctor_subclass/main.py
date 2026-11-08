class Square:
    def __init__(self, n):
        self.n = n

    def move(self):
        return self.n


class Sub(Square):
    def __init__(self):
        self.n = 2

    def move(self):
        return 9


sq = [Square(1)]
sq[0] = Sub()
s = sq[0]
assert s.move() == 9
