class Box:
    def __init__(self, v):
        self.v = v


b = Box(1.5)
a = Box(2)
assert type(a) is type(b)
