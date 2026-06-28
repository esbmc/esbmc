class Acc:
    def __init__(self, x: int):
        self.x = x


acc = Acc(10)
r = acc.x + 5
s = acc.x - 3
assert r == 16
assert s == 7
