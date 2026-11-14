class Box:
    def __init__(self, v):
        self.v = v

    def same(self, other):
        return isinstance(other, Box)


b = Box(1.5)
a = Box(2)
assert b.same(a)
