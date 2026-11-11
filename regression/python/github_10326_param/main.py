class A:
    pass


class Cell:
    def __init__(self, val):
        self.value = val + 23


try:
    c1 = Cell(A())
except TypeError:
    pass
c2 = Cell(2)
assert c2.value == 25
