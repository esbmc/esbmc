class S:
    def __init__(self, x: int, n: int):
        self.x = x
        self.n = n


s = S(3, 2)
assert (s.x << s.n) == 13
assert (s.x >> 1) == 1
