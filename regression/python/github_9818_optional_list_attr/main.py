class D:
    def __init__(self, e):
        self.cells = None if e else [0]

    def get(self, i):
        return self.cells[i]

d = D(False)
assert d.get(0) == 0
