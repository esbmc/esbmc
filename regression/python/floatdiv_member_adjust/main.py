class D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


d = D(7.0, 2.0)
assert d.x / d.y == 3.5
