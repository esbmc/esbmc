class Grid:
    def __init__(self) -> None:
        self.rows = [[1], [2, 2]]

    def promote(self) -> None:
        self.rows[0] = self.rows[1]


def run():
    g = Grid()
    g.promote()
    assert len(g.rows[0]) == 2
    assert sum(g.rows[0]) == 4

    g.rows[1] = [9, 9, 9]
    assert len(g.rows[1]) == 3


run()
