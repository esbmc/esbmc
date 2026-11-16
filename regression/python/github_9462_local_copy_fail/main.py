class D:
    def __init__(self):
        t = list(range(3))
        self.p = t


d = D()
assert d.p[2] == 1
