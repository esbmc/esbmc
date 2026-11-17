xs = [[1, 2], [3, 4]]
assert xs[1][0] == 3


class C:
    def __init__(self):
        self.v = [5, 6]


assert C().v[1] == 6
