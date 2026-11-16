def f(count: int, empty: bool):
    cells = [[0, 1, 2] for i in range(count)] if not empty else None
    cells[0].remove(1)
    assert cells[0] == [0, 2]
f(2, False)
