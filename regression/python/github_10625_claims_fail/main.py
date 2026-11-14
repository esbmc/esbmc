class Calc:
    def add(self, x, y):
        return x + y


c = Calc()
a = c.add(1, 2)
b = c.add(0.25, 0.25)
assert b < 0.4
