class F:
    def __init__(self, x: float):
        self.x = x


f = F(2.5)
r = f.x + 1.5
s = f.x * 2.0
t = f.x - 0.5
assert r == 4.0
assert s == 5.0
assert t == 2.0
