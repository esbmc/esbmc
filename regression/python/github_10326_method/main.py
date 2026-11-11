class A:
    pass


class Cell:
    def __init__(self, val):
        self.value = val

    def get(self):
        return self.value + 23


c1 = Cell(A())
c2 = Cell(2)
assert c2.get() == 25
