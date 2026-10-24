class P:
    def __init__(self, x: int):
        self.x = x


def larger(a, b):
    m = a.x if a.x > b.x else b.x
    return m


assert larger(P(3), P(2)) == 2
