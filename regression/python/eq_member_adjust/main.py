class P:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


p = P(5, 5)
assert p.a == p.b
assert p.a == 5
assert p.a != 7
