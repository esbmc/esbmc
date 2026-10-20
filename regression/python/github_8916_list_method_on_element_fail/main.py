def drop(cells, i, v):
    cells[i].remove(v)

cells = [[0, 1, 2], [0, 1, 2]]
drop(cells, 0, 1)
assert cells[0] == [0, 1]
