class S:

    def __init__(self, k):
        self.k = k

    def __call__(self, u):
        return self.k + u


class C:

    def __init__(self, splines):
        self.splines = splines

    def go(self):
        for spl in self.splines:
            assert spl(0) > 0


s = [S(1)][0]
assert s(0) == 1
C([S(1), S(-2)]).go()
