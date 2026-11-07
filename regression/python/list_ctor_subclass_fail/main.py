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



class Sub(Square):
    def __init__(self):
        self.n = 2


sq = [Square(1)]
sq[0] = Sub()
s = sq[0]
assert s.n == 1
