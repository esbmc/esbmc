class D:
    def __init__(self):
        self.cells = [0, 1, 2]

    def drop(self, v):
        self.cells.remove(v)

d = D()
d.drop(1)
assert d.cells == [0, 1]
