class A:
    pass


class Cell:
    def __init__(self, val):
        self.value = val


c1 = Cell(A())
c2 = Cell(2)
x = c2.value + 23
assert x == 25
