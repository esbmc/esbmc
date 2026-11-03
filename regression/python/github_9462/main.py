class D:
    def __init__(self, size):
        self.p = [i for i in range(size)]

    def find(self, i):
        return self.p[i]


d = D(3)
assert d.find(1) == 1
assert len(d.p) == 3
