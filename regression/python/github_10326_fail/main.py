class A:
    pass


class Cell:
    def __init__(self, val):
        self.value = val


c1 = Cell(A())
c2 = Cell(A())
x = c2.value + 23
