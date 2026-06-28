class Box:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


# Negative variant: `and` short-circuits to box.b (0), so the assert fails.
box = Box(1, 0)
r = box.a and box.b
assert r == 1
