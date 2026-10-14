class Line:

    def __init__(self, k):
        self.k = k

    def run(self, x):
        return x

    def __call__(self, t):
        return self.k + t


class Game:

    def __init__(self, lines):
        self.total = 0
        for f in lines:
            self.total += f(3)

    def run(self, lines):
        total = 0
        for f in lines:
            total += f(2)
        return total


lines = [Line(5)]
g = Game(lines)
assert g.total == 8
assert g.run(lines) == 8
