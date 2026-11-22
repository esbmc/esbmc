class Done:
    def __init__(self, count, empty=False):
        self.cells = None if empty else [[0, 1, 2] for i in range(count)]

    def remove(self, i, v):
        self.cells[i].remove(v)

d = Done(2)
d.remove(0, 1)
assert d.cells[0] == [0, 1, 2]
