class B:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


b = B(12, 10)
assert (b.x & b.y) == 8
assert (b.x | b.y) == 14
assert (b.x ^ b.y) == 6
