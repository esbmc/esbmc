class Calc:
    def add(self, x, y):
        return x + y


calcs = [Calc()]
assert calcs[0].add(1, 2) == 3
