class D:
    def __init__(self, e):
        self.cells = [3, 4] if e else None


d = D(True)
assert d.cells[1] == 4
assert len(d.cells) == 2
assert D(False).cells is None
