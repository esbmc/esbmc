def drop(cells, i, v):
    cells[i].remove(v)


def add(cells, i, v):
    cells[i].append(v)


class B:
    def __init__(self):
        self.cells = [[0, 1, 2], [0, 1, 2]]

    def drop(self, i: int, v: int):
        self.cells[i].remove(v)


cells = [[0, 1, 2], [0, 1, 2]]
drop(cells, 0, 1)
assert cells[0] == [0, 2]
add(cells, 1, 5)
assert cells[1] == [0, 1, 2, 5]

b = B()
b.drop(1, 2)
assert b.cells[1] == [0, 1]
