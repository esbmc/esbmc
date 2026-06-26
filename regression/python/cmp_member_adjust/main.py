class P:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


p = P(3, 9)
assert p.a < p.b
assert p.b > p.a
assert p.a <= 3
assert p.b >= 9
